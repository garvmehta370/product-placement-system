import os
import json
import base64
import requests
from PIL import Image, ImageDraw, ImageEnhance
from io import BytesIO
import numpy as np
import logging
import replicate
from openai import OpenAI
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
load_dotenv()

class ProductPlacementSystem:
    def __init__(self):
        self.replicate_client = replicate.Client(api_token=os.getenv('REPLICATE_API_TOKEN'))
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    def encode_image(self, image_path):
        """Convert image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def prepare_image_for_api(self, image):
        """Convert PIL Image to base64 data URL"""
        buffered = BytesIO()
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(buffered, format="JPEG", quality=85, optimize=True)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"
        
    def analyze_product_with_background(self, product_no_bg, background_image, product_placement):
        """Analyze product and background together using GPT-4 Vision to determine appropriate sizing"""
        logging.info("Analyzing product with background using GPT-4 Vision...")
        
        # Convert product image to base64
        product_buffered = BytesIO()
        if product_no_bg.mode != 'RGBA':
            product_no_bg = product_no_bg.convert('RGBA')
        product_no_bg.save(product_buffered, format="PNG")
        product_base64 = base64.b64encode(product_buffered.getvalue()).decode('utf-8')
        
        # Convert background image to base64
        bg_buffered = BytesIO()
        if background_image.mode != 'RGB':
            background_image = background_image.convert('RGB')
        background_image.save(bg_buffered, format="JPEG", quality=85)
        background_base64 = base64.b64encode(bg_buffered.getvalue()).decode('utf-8')

        prompt = f"""I have a product image and a background image. I want to place the product on/in the {product_placement} in the background.

        1. What is this product?
        2. Based on the background image size and context, what would be an appropriate percentage of the TOTAL background width for this product to occupy to look naturally sized and proportioned?
        3. Consider that this is for product placement advertising - the product should be clearly visible and appropriately sized relative to surrounding objects.

        IMPORTANT: For the width_percentage, be generous - products in advertisements are typically slightly larger than in real life to ensure visibility. If you're uncertain, err on the side of making the product more prominent (15-25% of image width is common for product placement).
        
        IMPORTANT: You must respond in valid JSON format with keys: 'product_name', 'recommended_width', 'width_percentage'. For 'width_percentage', return only the number (for example, if it's 15 percent, just return 15 and not 15%). DO NOT include any explanations outside the JSON structure."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{product_base64}"
                                }
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{background_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300,
                response_format={"type": "json_object"}  # Add this to explicitly request JSON
            )
            
            response_content = response.choices[0].message.content
            logging.info(f"GPT response: {response_content}")
            
            try:
                analysis = json.loads(response_content)
                return analysis
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse GPT response as JSON: {e}. Response was: {response_content}")
                # Try to extract JSON from text if it's embedded in explanations
                import re
                json_match = re.search(r'({.*})', response_content, re.DOTALL)
                if json_match:
                    try:
                        analysis = json.loads(json_match.group(1))
                        logging.info("Successfully extracted JSON from response text")
                        return analysis
                    except json.JSONDecodeError:
                        logging.error("Failed to extract JSON with regex")
                
                return {
                    "product_name": "product",
                    "recommended_width": 300,
                    "width_percentage": 20
                }
                
        except Exception as e:
            logging.error(f"GPT analysis failed: {str(e)}")
            return {
                "product_name": "product",
                "recommended_width": 300,
                "width_percentage": 20
            }
    
    def remove_background(self, image_path):
        """Remove background from product image using replicate API"""
        logging.info("Removing background from product image...")
        
        image_data_url = self.prepare_image_for_api(image_path)
        
        try:
            output = replicate.run(
                "851-labs/background-remover:a029dff38972b5fda4ec5d75d7d1cd25aeff621d2cf4946a41055d7db66b80bc",
                input={
                    "image": image_data_url,
                    "format": "png",
                    "reverse": False,
                    "threshold": 0,
                    "background_type": "rgba"
                }
            )
            
            response = requests.get(output)
            if response.status_code == 200:
                product_no_bg = Image.open(BytesIO(response.content))
                return product_no_bg
            else:
                raise Exception(f"Failed to download background-removed image: {response.status_code}")
        except Exception as e:
            logging.error(f"Background removal failed: {str(e)}")
            original = Image.open(image_path)
            return original.convert('RGBA')

    def enhance_product_image(self, product_image_path):
        """Enhance the product image to make text more readable using Real-ESRGAN"""
        logging.info("Enhancing product image to improve text readability...")
        
        image_data_url = f"data:image/jpeg;base64,{self.encode_image(product_image_path)}"
        
        try:
            # Using Real-ESRGAN model which is good for text enhancement
            output = replicate.run(
                "nightmareai/real-esrgan:42fed1c4974146d4d2414e2be2c5277c7fcf05fcc3a73abf41610695738c1d7b",
                input={
                    "image": image_data_url,
                    "scale": 4,
                    "face_enhance": False
                }
            )
            
            response = requests.get(output)
            if response.status_code == 200:
                enhanced_product = Image.open(BytesIO(response.content))
                # Save intermediate enhanced image for verification
                # enhanced_path = "enhanced_product.png"
                # enhanced_product.save(enhanced_path)
                logging.info(f"Enhanced product image")
                return enhanced_product
            else:
                raise Exception(f"Failed to download enhanced image: {response.status_code}")
        except Exception as e:
            logging.error(f"Product enhancement failed: {str(e)}")
            # If enhancement fails, return the original image
            return Image.open(product_image_path)

    def generate_background(self, background_prompt, product_placement):
        """Generate background image using Flux model"""
        logging.info("Generating background image...")
        enhanced_prompt = f"{background_prompt}, with {product_placement} prominently in the front, photorealistic, natural lightning, 8k quality"
        try:
            data = {
  "13": {
    "inputs": {
      "ckpt_name": "flux1-dev-fp8.safetensors"
    },
    "class_type": "CheckpointLoaderSimple",
    "_meta": {
      "title": "Load Checkpoint"
    }
  },
  "14": {
    "inputs": {
      "text": background_prompt,
      "clip": [
        "13",
        1
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "15": {
    "inputs": {
      "text": "(unnatural_limbs:1.8), incomplete limbs, mixed limbs",
      "clip": [
        "13",
        1
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "21": {
    "inputs": {
      "guidance": 3.5,
      "conditioning": [
        "14",
        0
      ]
    },
    "class_type": "FluxGuidance",
    "_meta": {
      "title": "FluxGuidance"
    }
  },
  "22": {
    "inputs": {
      "seed": 1024699332496720,
      "steps": 20,
      "cfg": 1,
      "sampler_name": "euler",
      "scheduler": "beta",
      "denoise": 1,
      "model": [
        "13",
        0
      ],
      "positive": [
        "21",
        0
      ],
      "negative": [
        "15",
        0
      ],
      "latent_image": [
        "84",
        0
      ]
    },
    "class_type": "KSampler",
    "_meta": {
      "title": "KSampler"
    }
  },
  "23": {
    "inputs": {
      "samples": [
        "22",
        0
      ],
      "vae": [
        "13",
        2
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "24": {
    "inputs": {
      "images": [
        "23",
        0
      ]
    },
    "class_type": "PreviewImage",
    "_meta": {
      "title": "Preview Image"
    }
  },
  "42": {
    "inputs": {},
    "class_type": "PreviewImage",
    "_meta": {
      "title": "Preview Image"
    }
  },
  "43": {
    "inputs": {},
    "class_type": "PreviewImage",
    "_meta": {
      "title": "Preview Image"
    }
  },
  "84": {
    "inputs": {
      "width": 1024,
      "height": 1024,
      "batch_size": 1
    },
    "class_type": "EmptyLatentImage",
    "_meta": {
      "title": "Empty Latent Image"
    }
  }
}
            json_string = json.dumps(data)
            output = replicate.run(
                "fofr/any-comfyui-workflow:ac793ee8fe34411d9cb3b0b3138152b6da8f7ebd178defaebe4b910ea3b16703",
                input={
                    "output_format": "webp",
                    "workflow_json": json_string,
                    "output_quality": 100,
                    "randomise_seeds": True,
                    "force_reset_cache": False,
                    "return_temp_files": True
                }
            )

            if output:
                image_url = str(output[0])
                logging.info(f"Generated Image: {image_url}")
                response = requests.get(image_url)
                if response.status_code == 200:
                    background = Image.open(BytesIO(response.content))
                    if background.mode != 'RGB':
                        background = background.convert('RGB')
                    return background
                else:
                    raise Exception(f"Failed to download background image: {response.status_code}")
            else:
                raise Exception("No output generated by Flux")
        except Exception as e:
            logging.error(f"Background generation failed: {str(e)}")
            raise

    def detect_placement_location(self, background_image, product_placement):
        logging.info(f"Detecting placement location for {product_placement}...")
        
        try:
            placement_query = "flat surface of the tabletop without legs" if "table" in product_placement.lower() else "countertop" if "counter" in product_placement.lower() else product_placement
            output = replicate.run(
                "adirik/grounding-dino:efd10a8ddc57ea28773327e881ce95e20cc1d734c589f7dd01d2036921ed78aa",
                input={
                    "image": self.prepare_image_for_api(background_image),
                    "query": placement_query,
                    "box_threshold": 0.05,
                    "text_threshold": 0.05,
                    "show_visualisation": False
                }
            )
            
            if isinstance(output, dict) and 'detections' in output:
                detections = output['detections']
                for detection in detections:
                    if isinstance(detection, dict) and 'bbox' in detection:
                        bbox = detection['bbox']
                        width = bbox[2] - bbox[0]
                        height = bbox[3] - bbox[1]
                        
                        if height < width * 0.5:
                            best_detection = bbox
                            break
            
            if not best_detection:
                logging.warning("No suitable placement location detected")
                w, h = background_image.size
                return [w*0.4, h*0.6, w*0.6, h*0.7]
    
            
            return best_detection
                
        except Exception as e:
            logging.error(f"Placement detection failed: {str(e)}")
            w, h = background_image.size
            return [w*0.4, h*0.6, w*0.6, h*0.7]

    def place_product(self, background, product, bbox, product_analysis):
        """Place product at detected location with bottom alignment"""
        logging.info("Placing product in background...")
        
        try:
            bg_width = background.size[0]
            x1, y1, x2, y2 = map(int, bbox)

            width_percentage = float(product_analysis['width_percentage'])
            scaling_factor = 1.25
            target_width = int(bg_width * (width_percentage / 100) * scaling_factor)

            product_width, product_height = product.size
            aspect_ratio = product_height / product_width
            target_height = int(target_width * aspect_ratio)
            
            product_resized = product.resize((target_width, target_height), Image.LANCZOS)
            
            bbox_center_x = (x1 + x2) // 2 
            bbox_center_y = (y1 + y2) // 2 
            bbox_center_yy = (bbox_center_y + y2) // 2
            bbox_center_yyy = (bbox_center_yy + bbox_center_y) // 2
            bbox_center_yyyy = (bbox_center_yyy + bbox_center_yy) // 2
            
            paste_x = bbox_center_x - (target_width // 2)
            paste_y = bbox_center_yyyy - target_height 
            
            surface_offset = 2 
            paste_y = paste_y - surface_offset
            
            paste_x = max(0, min(paste_x, bg_width - target_width))
            paste_y = max(0, min(paste_y, background.size[1] - target_height))
            
            if product_resized.mode != 'RGBA':
                product_resized = product_resized.convert('RGBA')
            
            alpha = product_resized.split()[3]
            
            feather_size = int(min(target_width, target_height) * 0.02)
            mask = Image.new('L', product_resized.size, 255)
            draw = ImageDraw.Draw(mask)
            
            for i in range(feather_size):
                opacity = int(255 * (i / feather_size))
                draw.rectangle(
                    [i, i, target_width-i, target_height-i],
                    outline=opacity,
                    width=1
                )
            
            final_mask = Image.fromarray(
                np.minimum(
                    np.array(alpha),
                    np.array(mask)
                ).astype(np.uint8)
            )
            
            result = background.copy()
            
            enhancer = ImageEnhance.Brightness(product_resized)
            product_resized = enhancer.enhance(1.05)
            
            result.paste(product_resized, (paste_x, paste_y), final_mask)
            
            return result
            
        except Exception as e:
            logging.error(f"Product placement failed: {str(e)}")
            return background

    def upload_to_imgbb(self, image_path):
        """Upload an image to imgbb and return the URL"""
        try:
            api_key = os.getenv('IMGBB_API_KEY')
            if not api_key:
                logging.error("IMGBB_API_KEY environment variable not set")
                return None

            with open(image_path, "rb") as file:
                url = "https://api.imgbb.com/1/upload"
                payload = {
                    "key": api_key,
                    "image": base64.b64encode(file.read()),
                }
                response = requests.post(url, payload)
                
                if response.status_code == 200:
                    json_data = response.json()
                    if json_data.get("success", False):
                        image_url = json_data["data"]["url"]
                        logging.info(f"Image uploaded successfully: {image_url}")
                        return image_url
                    else:
                        logging.error(f"Upload failed: {json_data.get('error', {}).get('message', 'Unknown error')}")
                else:
                    logging.error(f"Upload failed with status code: {response.status_code}")
                    
                return None
        except Exception as e:
            logging.error(f"Image upload failed: {str(e)}")
            return None
        
    def process_image(self, product_image_path, background_prompt, product_placement):
        """Main processing pipeline"""
        try:
            enhanced_product = self.enhance_product_image(product_image_path)

            product_no_bg = self.remove_background(enhanced_product)
            #product_no_bg = self.remove_background(product_image_path)
            
            background = self.generate_background(background_prompt, product_placement)

            product_analysis = self.analyze_product_with_background(
                product_no_bg,
                background,
                product_placement
            )
            
            bbox = self.detect_placement_location(background, product_placement)
            
            final_image = self.place_product(background, product_no_bg, bbox, product_analysis)
            
            final_path = "final_output.png"
            final_image.save(final_path)
            
            image_url = self.upload_to_imgbb(final_path)
            
            logging.info("Image processing completed successfully")
            return final_image, image_url
            
        except Exception as e:
            logging.error(f"Image processing failed: {str(e)}")
            raise

def main():
    system = ProductPlacementSystem()
    
    product_image_path = "product5.jpeg"
    #background_prompt = "modern minimalist kitchen with marble countertops, natural lighting"
    #product_placement = "countertop"
    background_prompt = "A modern Indian kitchen with a warm and inviting ambiance. The scene is captured from the center of a sleek wooden breakfast table, perfectly framing the composition. An Indian mother and her young son (aged 5-7 years) sitting across from each other at a wooden dining table in a modern, well-lit kitchen. They are enjoying steaming bowls of masala-flavored oats, smiling warmly and engaging with each other. They have distinct Indian facial features, warm brown skin tones, dark hair, and are dressed in casual yet traditional Indian or modern clothing.  Their body language is natural, relaxed, and joyful. The stylish kitchen in the background features elegant cabinets, a well-organized countertop, and soft natural light streaming in from a large window. The composition ensures the central area remains visually clear, maintaining balance and harmony in the scene.\n\n\n\n\n\n"
    product_placement = "table"
    
    try:
        final_image, image_url = system.process_image(product_image_path, background_prompt, product_placement)
        if image_url:
            logging.info(f"Final image available at: {image_url}")
        else:
            logging.info("Image was processed but could not be uploaded to hosting service")
    except Exception as e:
        logging.error(f"Process failed: {str(e)}")

if __name__ == "__main__":
    main()