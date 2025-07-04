import perception_encoder.pe as meta_pe

def get_perception_encoder(model_name):
    model = meta_pe.VisionTransformer.from_config(model_name, pretrained=True)
    return model

if __name__ == "__main__":
    import torch
    from PIL import Image
    from torchvision.transforms import ToTensor

    # PE configs: ['PE-Core-G14-448', 'PE-Core-L14-336', 'PE-Core-B16-224', 'PE-Lang-G14-448', 'PE-Lang-L14-448', 'PE-Spatial-G14-448']
    print("PE configs:", meta_pe.VisionTransformer.available_configs())
    
    # model = get_perception_encoder(model_name="PE-Lang-L14-448")
    model = get_perception_encoder(model_name="PE-Spatial-G14-448")
    model.eval()
    model.cuda()
    out = model.forward_features(torch.rand(1, 3, 448, 448).cuda())
    print(out.shape)
    
    import time
    time.sleep(10)