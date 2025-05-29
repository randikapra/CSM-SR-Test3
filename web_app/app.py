from flask import Flask, render_template, request, jsonify, send_file
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
import time
from werkzeug.utils import secure_filename
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Global variables for models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models = {}
lpips_loss = lpips.LPIPS(net='alex').to(device)

# Model configurations
MODEL_CONFIGS = {
    'your_model': {
        'name': 'Structure-Informed SR',
        'path': 'path/to/your/model.pth',
        'description': 'Your novel structure-informed super-resolution model'
    },
    'vdsr': {
        'name': 'VDSR',
        'path': 'path/to/vdsr_model.pth',
        'description': 'Very Deep Super Resolution baseline'
    },
    'esrgan': {
        'name': 'ESRGAN',
        'path': 'path/to/esrgan_model.pth',
        'description': 'Enhanced Super Resolution GAN'
    }
}

def load_models():
    """Load all pre-trained models"""
    global models
    try:
        for model_key, config in MODEL_CONFIGS.items():
            if os.path.exists(config['path']):
                # Replace this with your actual model loading logic
                # models[model_key] = torch.load(config['path'], map_location=device)
                # models[model_key].eval()
                pass
        print(f"Loaded {len(models)} models successfully")
    except Exception as e:
        print(f"Error loading models: {e}")

def preprocess_image(image_path):
    """Preprocess image for model input"""
    image = Image.open(image_path).convert('RGB')
    
    # Resize if too large (optional)
    max_size = 512
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    # Convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor, image

def super_resolve_image(input_tensor, model_name='your_model'):
    """Apply super-resolution using specified model"""
    try:
        # Replace this with your actual model inference
        # For demo purposes, we'll simulate super-resolution
        with torch.no_grad():
            # Simulated super-resolution (replace with actual model)
            # output = models[model_name](input_tensor)
            
            # For demo: simple upscaling simulation
            upscale_factor = 2
            output = torch.nn.functional.interpolate(
                input_tensor, 
                scale_factor=upscale_factor, 
                mode='bicubic', 
                align_corners=False
            )
            
        return output
    except Exception as e:
        print(f"Error in super-resolution: {e}")
        return input_tensor

def calculate_metrics(original, enhanced, ground_truth=None):
    """Calculate PSNR, SSIM, and LPIPS metrics"""
    metrics = {}
    
    try:
        # Convert tensors to numpy arrays
        if torch.is_tensor(original):
            orig_np = original.squeeze().cpu().numpy().transpose(1, 2, 0)
        else:
            orig_np = np.array(original)
            
        if torch.is_tensor(enhanced):
            enh_np = enhanced.squeeze().cpu().numpy().transpose(1, 2, 0)
        else:
            enh_np = np.array(enhanced)
        
        # Ensure values are in [0, 1] range
        orig_np = np.clip(orig_np, 0, 1)
        enh_np = np.clip(enh_np, 0, 1)
        
        # Calculate PSNR and SSIM
        if ground_truth is not None:
            gt_np = np.array(ground_truth)
            gt_np = np.clip(gt_np, 0, 1)
            
            metrics['psnr'] = float(peak_signal_noise_ratio(gt_np, enh_np))
            metrics['ssim'] = float(structural_similarity(gt_np, enh_np, multichannel=True))
            
            # Calculate LPIPS
            gt_tensor = torch.from_numpy(gt_np.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
            enh_tensor = torch.from_numpy(enh_np.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
            metrics['lpips'] = float(lpips_loss(gt_tensor * 2 - 1, enh_tensor * 2 - 1))
        else:
            # Compare with bicubic upscaling as baseline
            bicubic = cv2.resize(orig_np, (enh_np.shape[1], enh_np.shape[0]), interpolation=cv2.INTER_CUBIC)
            metrics['psnr'] = float(peak_signal_noise_ratio(bicubic, enh_np))
            metrics['ssim'] = float(structural_similarity(bicubic, enh_np, multichannel=True))
            
            # LPIPS comparison
            bicubic_tensor = torch.from_numpy(bicubic.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
            enh_tensor = torch.from_numpy(enh_np.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
            metrics['lpips'] = float(lpips_loss(bicubic_tensor * 2 - 1, enh_tensor * 2 - 1))
            
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        metrics = {'psnr': 0.0, 'ssim': 0.0, 'lpips': 1.0}
    
    return metrics

def tensor_to_base64(tensor):
    """Convert tensor to base64 string for web display"""
    try:
        # Convert tensor to PIL Image
        tensor_cpu = tensor.squeeze().cpu()
        if tensor_cpu.dim() == 3:
            tensor_cpu = tensor_cpu.clamp(0, 1)
            array = (tensor_cpu.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        else:
            array = (tensor_cpu.numpy() * 255).astype(np.uint8)
        
        image = Image.fromarray(array)
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        
        return base64.b64encode(buffer.getvalue()).decode()
    except Exception as e:
        print(f"Error converting tensor to base64: {e}")
        return ""

@app.route('/')
def index():
    return render_template('index.html', models=MODEL_CONFIGS)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    model_name = request.form.get('model', 'your_model')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the image
            start_time = time.time()
            
            # Preprocess
            input_tensor, original_image = preprocess_image(filepath)
            
            # Super-resolve
            output_tensor = super_resolve_image(input_tensor, model_name)
            
            processing_time = time.time() - start_time
            
            # Calculate metrics
            metrics = calculate_metrics(input_tensor, output_tensor)
            
            # Convert results to base64 for display
            original_b64 = tensor_to_base64(input_tensor)
            enhanced_b64 = tensor_to_base64(output_tensor)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'original_image': original_b64,
                'enhanced_image': enhanced_b64,
                'metrics': {
                    'psnr': round(metrics['psnr'], 2),
                    'ssim': round(metrics['ssim'], 4),
                    'lpips': round(metrics['lpips'], 4)
                },
                'processing_time': round(processing_time, 2),
                'model_used': MODEL_CONFIGS[model_name]['name']
            })
            
        except Exception as e:
            return jsonify({'error': f'Processing failed: {str(e)}'})
    
    return jsonify({'error': 'Invalid file format'})

@app.route('/compare_models', methods=['POST'])
def compare_models():
    """Compare results across different models"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess once
        input_tensor, original_image = preprocess_image(filepath)
        
        results = {}
        
        # Process with each available model
        for model_key in MODEL_CONFIGS.keys():
            start_time = time.time()
            output_tensor = super_resolve_image(input_tensor, model_key)
            processing_time = time.time() - start_time
            
            metrics = calculate_metrics(input_tensor, output_tensor)
            
            results[model_key] = {
                'name': MODEL_CONFIGS[model_key]['name'],
                'enhanced_image': tensor_to_base64(output_tensor),
                'metrics': {
                    'psnr': round(metrics['psnr'], 2),
                    'ssim': round(metrics['ssim'], 4),
                    'lpips': round(metrics['lpips'], 4)
                },
                'processing_time': round(processing_time, 2)
            }
        
        # Clean up
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'original_image': tensor_to_base64(input_tensor),
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': f'Comparison failed: {str(e)}'})

@app.route('/system_info')
def system_info():
    """Return system information"""
    return jsonify({
        'device': str(device),
        'cuda_available': torch.cuda.is_available(),
        'models_loaded': len(models),
        'supported_formats': ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    })

if __name__ == '__main__':
    print("Loading models...")
    load_models()
    print("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5000)