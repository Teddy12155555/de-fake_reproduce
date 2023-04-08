import pickle
from tqdm import tqdm
from diffusers import StableDiffusionPipeline

if __name__ == '__main__':
    pkl_path = '../word_dict.pickle'
    image_path = 'SD+MSCOCO/Fake/'
    
    with open(pkl_path, 'rb') as f:
        word_dict = pickle.load(f)
        
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to('cuda')
    
    for k, v in tqdm(word_dict.items()):
        pre_img = pipe(v).images[0]
        pre_img.save(f'{image_path}{k}.png')