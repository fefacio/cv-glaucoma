import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer

        # Armazena os mapas de ativação e gradientes
        self.activations = None
        self.gradients = None

        # Hook: salva ativações e gradientes
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # Registra os hooks
        self.fwd = target_layer.register_forward_hook(forward_hook)
        self.bwd = target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        # Forward
        output = self.model(input_tensor)

        # Se for binário (1 saída), assume a classe positiva
        if class_idx is None:
            class_idx = 0

        # Zera gradientes antigos
        self.model.zero_grad()

        # Backward pass para o score da classe alvo
        target = output[:, class_idx]
        target.backward()

        # Obtém média dos gradientes no espaço HxW (global average pooling)
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # Multiplica os gradientes pelas ativações
        activations = self.activations[0]
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]

        # Soma os canais (dim=0) e aplica ReLU
        heatmap = torch.sum(activations, dim=0)
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)  # Normaliza entre 0 e 1

        return heatmap.cpu().numpy()

    def remove_hooks(self):
        self.fwd.remove()
        self.bwd.remove()



# Suponha que você tenha:
# model: sua ResNet50 fine-tunada
# image_tensor: tensor de shape (1, 3, H, W) já preprocessado

# gradcam = GradCAM(model, model.layer4)

# # Gera heatmap
# heatmap = gradcam.generate(image_tensor)

# # Exibe o resultado
# plt.imshow(heatmap, cmap='jet')
# plt.colorbar()
# plt.title("Grad-CAM")
# plt.show()

# gradcam.remove_hooks()
