from PIL import Image
from base_classes import Embedder
import torch
from torchvision import transforms

class ImageEmbedder(Embedder):
    """Embedder using MobileNet for image embeddings."""
    def __init__(self, model_name='mobilenet_v2', device='cpu'):
        super().__init__(model_name)
        self.device = device
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        self.model = self.model.to(device)
        self.model.eval()  # Set to evaluation mode
        
        # Remove the classification head to get embeddings
        self.model.classifier = torch.nn.Identity()
        
        # Image preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
    def embed(self, image_path):
        """Compute embedding for a single image."""
        try:
            # Load and preprocess image
            import pdb ; pdb.set_trace()
            image = Image.open(image_path,'rb').convert('RGB')
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Generate embedding
            with torch.no_grad():
                embedding = self.model(image_tensor)
            
            return embedding.squeeze().cpu().numpy()
        except Exception as e:
            logging.error(f"Error embedding image {image_path}: {e}")
            return None
            
    def embed_batch(self, image_paths):
        """Compute embeddings for a batch of images."""
        embeddings = []
        for path in image_paths:
            embedding = self.embed(path)
            if embedding is not None:
                embeddings.append(embedding)
        return np.stack(embeddings) if embeddings else None 
