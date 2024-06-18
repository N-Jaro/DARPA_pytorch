import torch
from models.unetTransformer import U_Transformer, U_Transformer_Lightning
from models.lit_training import LitTraining



# # Initialize model
model = U_Transformer_Lightning.load_from_checkpoint(in_channels=6, classes=1, checkpoint_path="/projects/bcxi/nathanj/DARPA_pytorch/checkpoints/utransformer_experiment_pl_20240618_121333/Utransformer-epoch=00-val_loss=0.84.ckpt")

# # Initialize Lightning model
# lit_model = LitTraining.load_from_checkpoint(U_Transformer(in_channels=6, classes=1), checkpoint_path="/projects/bcxi/nathanj/DARPA_pytorch/checkpoints/Utransformer/best_model.ckpt")

# model = LitTraining.load_from_checkpoint(model,checkpoint_path="/projects/bcxi/nathanj/DARPA_pytorch/checkpoints/Utransformer/best_model.ckpt")