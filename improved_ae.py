import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from guided_diffusion.unet import ResBlock, AttentionBlock, Downsample, Upsample, conv_nd, normalization
import os
from guided_diffusion.bratsloader import BRATSDataset
from tqdm import tqdm
from timm.utils import ModelEmaV3
import wandb

import torch
import os.path
import nibabel

wandb.login(key="b3990523b6efe11b984fad8a8e64911f78399735") 

class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, directory, test_flag=False):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_001_XXX_123_w.nii.gz
                  where XXX is one of t1, t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        '''
        super().__init__()
        self.directory = os.path.expanduser(directory)

        self.test_flag=test_flag
        if test_flag:
            self.seqtypes = ['t2', 'flair']
        else:
            self.seqtypes = ['t2', 'flair', 'seg']

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have data
            if not dirs:
                files.sort()
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    # print("Extracting: ", f)
                    if(len(f.split('_'))<4):
                        continue
                    seqtype = f.split('_')[3]
                    datapoint[seqtype] = os.path.join(root, f)
                assert set(['t2', 'flair', 'seg']) == self.seqtypes_set, \
                    f'datapoint is incomplete, keys are {datapoint.keys()}'
                self.database.append(datapoint)

    def __getitem__(self, x):
        out = []
        filedict = self.database[x]
        for seqtype in self.seqtypes:
            # print(filedict['t1'])
            number=filedict['t1'].split('/')[-2]
            nib_img = nibabel.load(filedict[seqtype])
            out.append(torch.tensor(nib_img.get_fdata()))
        out = torch.stack(out)
        out_dict = {}
        if self.test_flag:
            # change path here according to the data's path
            path2 = './brats-data/test_labels/' + str(
                number) + '-label.nii'


            seg=nibabel.load(path2)
            seg=seg.get_fdata()
            image = torch.zeros(2, 256, 256)
            image[:, 8:-8, 8:-8] = out
            label = seg[None, ...]
            if seg.max() > 0:
                weak_label = 1
            else:
                weak_label = 0
            out_dict["y"]=weak_label
        else:
            image = torch.zeros(2,256,256)
            image[:,8:-8,8:-8]=out[:-1,...]		#pad to a size of (256,256)
            label = out[-1, ...][None, ...]
            if label.max()>0:
                weak_label=1
            else:
                weak_label=0
            out_dict["y"] = weak_label

        return (image, out_dict, weak_label, label, number )

    def __len__(self):
        return len(self.database)
    

class AEEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        model_channels: int = 128,
        channel_mult: tuple = (1, 1, 2, 2, 4, 4),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        dims: int = 2,
        conv_resample: bool = True,
        use_checkpoint: bool = False,
        use_scale_shift_norm: bool = False,
        num_heads: int = 1,
        use_new_attention_order: bool = False,
    ):
        super().__init__()
        self.model_channels = model_channels
        self.time_embed_dim = model_channels * 4
        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks
        self.dims = dims
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        self.num_head_channels = -1
        self.use_new_attention_order = use_new_attention_order
        
        # Create a dummy embedding
        self.dummy_emb = nn.Parameter(th.zeros(1, self.time_embed_dim), requires_grad=False)
        
        self.initial = nn.Sequential(
            conv_nd(dims, in_channels, model_channels, 3, padding=1),
            normalization(model_channels),
            nn.SiLU()
        )
        self.resblocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = model_channels
        for level, mult in enumerate(channel_mult):
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_blocks.append(
                    ResBlock(
                        ch,
                        emb_channels=self.time_embed_dim,
                        out_channels=mult * model_channels,
                        dropout=dropout,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                )
                ch = mult * model_channels
            self.resblocks.append(level_blocks)
            if level != len(channel_mult) - 1:
                self.downsamples.append(
                    Downsample(ch, conv_resample, dims=dims, out_channels=ch)
                )
        self.middle = nn.ModuleList([
            ResBlock(ch, emb_channels=self.time_embed_dim, dropout=dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=self.num_head_channels, use_new_attention_order=use_new_attention_order),
            ResBlock(ch, emb_channels=self.time_embed_dim, dropout=dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm),
        ])
        
    def forward(self, x):
        print(f"Input shape: {x.shape}")
        features = {}
        # print(f"Input shape: {x.shape}")
        
        h = self.initial(x)
        # print(f"After initial shape: {h.shape}")
        
        # Create a batch of dummy embeddings
        emb = self.dummy_emb.expand(x.shape[0], -1)
        
        # Pass through ResBlocks and downsampling layers
        for i, level_blocks in enumerate(self.resblocks):
            # print(f"Level {i} starting shape: {h.shape}")
            for j, block in enumerate(level_blocks):
                h = block(h, emb=emb)
                # print(f"Level {i}, block {j}, shape after ResBlock: {h.shape}")
            res = 256 // (2 ** i)
            features[res] = h
            # print(f"Saving feature[{res}] with shape: {h.shape}")
            if i != len(self.resblocks) - 1:
                h = self.downsamples[i](h)
                # print(f"After downsample {i}, shape: {h.shape}")
        
        # Handle the middle layers
        # print(f"Before middle blocks shape: {h.shape}")
        for i, layer in enumerate(self.middle):
            if isinstance(layer, ResBlock):
                h = layer(h, emb=emb)
            else:
                h = layer(h)
            # print(f"After middle block {i}, shape: {h.shape}")
        
        features["bn"] = h  # Add the bottleneck features
        # print(f"Bottleneck feature[8] shape: {h.shape}")
        
        # print("All feature shapes:")
        for k in features.keys():
            print(f"feature[{k}]: {features[k].shape}")
            
        return features
class Decoder(nn.Module):
    def __init__(self, base_channels=128, out_channels=2):
        super().__init__()
        self.base_channels = base_channels
        
        # Create a dummy embedding of zeros
        self.time_embed_dim = base_channels * 4
        self.dummy_emb = nn.Parameter(th.zeros(1, self.time_embed_dim), requires_grad=False)
        
        # Start with bottleneck (8x8) features
        # Upsampling from 8→16
        self.up1 = Upsample(base_channels * 4, True, dims=2, out_channels=base_channels * 4)
        self.block5 = ResBlock(base_channels * 4, emb_channels=self.time_embed_dim, 
                               out_channels=base_channels * 4, dropout=0.0)
        
        # Upsampling from 16→32
        self.up2 = Upsample(base_channels * 4, True, dims=2, out_channels=base_channels * 3)
        self.block4 = ResBlock(base_channels * 3, emb_channels=self.time_embed_dim, 
                               out_channels=base_channels * 3, dropout=0.0)
        
        # Upsampling from 32→64
        self.up3 = Upsample(base_channels * 3, True, dims=2, out_channels=base_channels * 2)
        self.block3 = ResBlock(base_channels * 2, emb_channels=self.time_embed_dim, 
                               out_channels=base_channels * 2, dropout=0.0)
        
        # Upsampling from 64→128
        self.up4 = Upsample(base_channels * 2, True, dims=2, out_channels=base_channels * 1)
        self.block2 = ResBlock(base_channels * 1, emb_channels=self.time_embed_dim, 
                               out_channels=base_channels * 1, dropout=0.0)
        
        # Upsampling from 128→256
        self.up5 = Upsample(base_channels, True, dims=2, out_channels=base_channels)
        self.block1 = ResBlock(base_channels, emb_channels=self.time_embed_dim, 
                               out_channels=base_channels, dropout=0.0)
        
        # Final convolution to get the output channels
        self.final = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1),
            # nn.Tanh()
            nn.Sigmoid()
        )
    
    def forward(self, features):
        # Create a batch of dummy embeddings
        emb = self.dummy_emb.expand(features["bn"].shape[0], -1)
        
        # Start with bottleneck features only
        x = features[8]  # Shape should be [batch, base_channels*4, 8, 8]
        
        # Upsample to 16x16
        x = self.up1(x)  # Shape becomes [batch, base_channels*4, 16, 16]
        x = self.block5(x, emb=emb)  # No concatenation
        
        # Upsample to 32x32
        x = self.up2(x)  # Shape becomes [batch, base_channels*3, 32, 32]
        x = self.block4(x, emb=emb)  # No concatenation
        
        # Upsample to 64x64
        x = self.up3(x)  # Shape becomes [batch, base_channels*2, 64, 64]
        x = self.block3(x, emb=emb)  # No concatenation
        
        # Upsample to 128x128
        x = self.up4(x)  # Shape becomes [batch, base_channels, 128, 128]
        x = self.block2(x, emb=emb)  # No concatenation
        
        # Upsample to 256x256
        x = self.up5(x)  # Shape becomes [batch, base_channels, 256, 256]
        x = self.block1(x, emb=emb)  # No concatenation
        
        # Final convolution
        x = self.final(x)  # Shape becomes [batch, out_channels, 256, 256]

        print(f"Final output shape for this batch: {x.shape}")
        
        return x
        
import matplotlib.pyplot as plt

def plot_reconstruction(images, recons, epoch, recon_path, model_name):
    n = images.shape[0]

    fig, axes = plt.subplots(2, n, figsize=(n*3, 6))
    fig.suptitle(f"recons_e{epoch}_{model_name}")
    for i in range(n):
        # Original
        axes[0, i].imshow(images[i].mean(axis=0), cmap='gray')  
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')
        # Reconstruction
        axes[1, i].imshow(recons[i].mean(axis=0) , cmap='gray') 
        axes[1, i].set_title("Reconstruction")
        axes[1, i].axis('off')
    plt.tight_layout()
    
    plt.savefig(f"{recon_path}/recons_e{epoch}_{model_name}.png", bbox_inches='tight')  # Save as PNG
    wandb.log({f"recons_e{epoch}": wandb.Image(plt)})
    plt.close()


def train_autoencoder(
    device="cuda",
    epochs=200, 
    save_path="checkpoints/", 
    eval_freq=1,
    save_freq=10, 
    batch_size=16,
    lr=1e-4,
    ema_decay=0.999,
    patience = 5
):
    # Initialize wandb for logging
    wandb.init(project="improved_ae_training", config={
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "ema_decay": ema_decay,
        "architecture": "BrainAutoencoder",
        "optimizer": "Adam",
        "scheduler": "ReduceLROnPlateau"
    })

    os.makedirs(save_path, exist_ok=True)
    recon_path = f"{save_path}/unhealthy21_reconstructions"
    os.makedirs(recon_path, exist_ok=True)

    encoder = AEEncoder(in_channels=2).to(device)
    decoder = Decoder(out_channels=2).to(device)

    ema_encoder = ModelEmaV3(encoder, decay=ema_decay, device=device)
    ema_decoder = ModelEmaV3(decoder, decay=ema_decay, device=device)

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    # Changed to ReduceLROnPlateau like the reference code
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                   patience=5, verbose=True)
    criterion = nn.MSELoss()

    # Load Dataset (keeping your existing dataset)
    dataset = BRATSDataset("/home/cs23m1009/Siddesh/healthy", test_flag=False)
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Print training information
    print(f"Training on {len(train_ds)} slices, validating on {len(val_ds)} slices")
    print(f"Initial learning rate: {optimizer.param_groups[0]['lr']}")
    
    # Track best models
    best_val_loss = float("inf")
    best_ema_val_loss = float("inf")

    # counter for early stopping
    low_vl_counter = 0

    def validate(enc, dec, model_name="Regular"):
        enc.eval()
        dec.eval()
        val_loss = 0.0
        
        with th.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"[Validating {model_name}]", leave=False)):
                x = batch[0].to(device)
                
                # Forward pass
                features = enc(x)
                recon = dec(features)
                
                # Compute loss
                loss = criterion(recon, x)
                val_loss += loss.item()
                
                # Save visualization for the first batch
                if batch_idx == 0:
                    plot_reconstruction(x.cpu(), recon.cpu(), epoch, recon_path, model_name)
        
        # Average loss
        avg_val_loss = val_loss / len(val_loader)
        print(f"{model_name} Validation Loss: {avg_val_loss:.6f}")
        
        return avg_val_loss, x, recon

    # train
    for epoch in range(1, epochs + 1):
        encoder.train()
        decoder.train()
        total_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"[Epoch {epoch}/{epochs}] Training", leave=False)
        for batch_idx, batch in enumerate(train_bar):
            x = batch[0].to(device)
            
            # Debug input/output ranges on first batch
            if batch_idx == 0:
                print(f"Input range: {x.min().item():.4f} to {x.max().item():.4f}")
            
            # Forward pass
            features = encoder(x)
            x_recon = decoder(features)
            loss = criterion(x_recon, x)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()), 
                max_norm=1.0
            )
            optimizer.step()

            # Update EMA models
            ema_encoder.update(encoder)
            ema_decoder.update(decoder)

            # Track loss
            total_loss += loss.item()
            
            # Debug output range on first batch
            if batch_idx == 0:
                print(f"Output range: {x_recon.min().item():.4f} to {x_recon.max().item():.4f}")
            
            # Update progress bar
            train_bar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
            
            # Log periodically to wandb (similar to reference code)
            if batch_idx % 10 == 0:
                wandb.log({
                    "train_loss_step": loss.item(),
                    "epoch": epoch,
                    "batch": batch_idx,
                    "learning_rate": optimizer.param_groups[0]['lr']
                })

        # Calculate average training loss
        avg_train_loss = total_loss / len(train_loader)
        
        avg_val_loss, _, _ = validate(encoder, decoder, "REG")
        avg_ema_val_loss, _, _ = validate(ema_encoder.module, ema_decoder.module, "EMA")
        
        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)
        print(f"Learning rate after epoch {epoch}: {optimizer.param_groups[0]['lr']}")
        
        # Log metrics to wandb
        wandb.log({
            "train_loss_epoch": avg_train_loss,
            "val_loss": avg_val_loss,
            "ema_val_loss": avg_ema_val_loss,
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        print(f"[Epoch {epoch}/{epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | EMA Val: {avg_ema_val_loss:.4f}")

        # Save best regular model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            th.save({
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "ema_encoder": ema_encoder.state_dict(),
                "ema_decoder": ema_decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "val_loss": avg_val_loss
            }, f"{save_path}/best_checkpoint.pth")
            print(f"Saved new best model with validation loss: {avg_val_loss:.6f}")
            low_vl_counter = 0
        else:
            low_vl_counter += 1
            if low_vl_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
                
        # Save best EMA model
        if avg_ema_val_loss < best_ema_val_loss:
            best_ema_val_loss = avg_ema_val_loss
            th.save({
                "encoder": ema_encoder.state_dict(),
                "decoder": ema_decoder.state_dict(),
                "epoch": epoch,
                "val_loss": avg_ema_val_loss
            }, f"{save_path}/best_ema_checkpoint.pth")
            print(f"Saved new best EMA model with validation loss: {avg_ema_val_loss:.6f}")

        # Periodic save
        if epoch % save_freq == 0:
            th.save({
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "ema_encoder": ema_encoder.state_dict(),
                "ema_decoder": ema_decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "val_loss": avg_val_loss,
                "ema_val_loss": avg_ema_val_loss
            }, f"{save_path}/checkpoint_epoch{epoch}.pth")
            print(f"Saved checkpoint at epoch {epoch}")

    # Save final model
    th.save({
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "ema_encoder": ema_encoder.state_dict(),
        "ema_decoder": ema_decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epochs,
        "val_loss": avg_val_loss,
        "ema_val_loss": avg_ema_val_loss
    }, f"{save_path}/final_checkpoint.pth")
    
    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    DEVICE = th.device("cuda" if th.cuda.is_available() else "cpu")
    print("Device: ", DEVICE)

    train_autoencoder(
        device=DEVICE,
        epochs=200,
        save_path="sid_checkpoints/",
        eval_freq=1,    # eval after num epochs
        save_freq=5,    # save ckpt after num epochs
        batch_size=12,
        lr=1e-4,
        ema_decay=0.999,
        patience = 15
    )








# # Load ckpt ###############
#     def plot_unhealthy_reconstruction(images, recons, epoch, recon_path, batch_idx):
#         print(images.max(), images.min())
#         # with torch.no_grad():
#         #     images = batch.to(device)
#         #     recons = ae_model(images)
#         # images = images.cpu()
#         # recon = recon.cpu()
#         n = images.shape[0]

#         fig, axes = plt.subplots(2, n, figsize=(n*3, 6))
#         for i in range(n):
#             # Original
#             axes[0, i].imshow(images[i].mean(axis=0), cmap='gray')  # or 'gray'
#             axes[0, i].set_title("Original")
#             axes[0, i].axis('off')
#             # Reconstruction
#             axes[1, i].imshow(recons[i].mean(axis=0) , cmap='gray')  # or 'gray'
#             axes[1, i].set_title("Reconstruction")
#             axes[1, i].axis('off')
#         plt.tight_layout()
        
#         plt.savefig(f"{recon_path}/reconstruction_e{epoch}_{batch_idx}.png", bbox_inches='tight')  # Save as PNG
#         # wandb.log({f"reconstruction_e{epoch}": wandb.Image(plt)})
#         plt.close()

#     dataset = BRATSDataset("/home/cs23m1009/projects/diff-anomaly/OUTPUTS/data/testing21", test_flag=False)  # /home/cs23m1009/Siddesh/healthy
#     val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

#     checkpoint = th.load(f"/home/cs23m1009/projects/diff-anomaly/sid_checkpoints2/checkpoint_epoch20.pth")

#     encoder.load_state_dict(checkpoint["encoder"])
#     decoder.load_state_dict(checkpoint["decoder"])
#     ema_encoder.load_state_dict(checkpoint["ema_encoder"])
#     ema_decoder.load_state_dict(checkpoint["ema_decoder"])
#     epoch = checkpoint["epoch"]
#     val_loss = checkpoint["val_loss"]

#     encoder.eval()
#     decoder.eval()
#     val_loss, ema_val_loss = 0.0, 0.0

#     with th.no_grad():
#         for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"[Validation]", leave=False)):
#             print(batch[0].shape)
#             x = batch[0].to(device)

#             # Normal
#             recon = decoder(encoder(x))
#             val_loss += criterion(recon, x).item()

#             # EMA
#             recon_ema = ema_decoder.module(ema_encoder.module(x))
#             ema_val_loss += criterion(recon_ema, x).item()
#             plot_unhealthy_reconstruction(x.cpu(), recon.cpu(), '20', recon_path, batch_idx)

#     avg_val_loss = val_loss / len(val_loader)
#     avg_ema_val_loss = ema_val_loss / len(val_loader)

#     # # Log validation metrics to wandb
#     # wandb.log({
#     #     "val_loss": avg_val_loss,
#     #     "ema_val_loss": avg_ema_val_loss,
#     #     "epoch": epoch
#     # })

#     print(f"[Unhealthy] Val Loss: {avg_val_loss:.4f} | EMA Val: {avg_ema_val_loss:.4f}")

#     # #########
#     return
# -----final------