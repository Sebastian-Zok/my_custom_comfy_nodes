import torch
import numpy as np
from skimage.measure import label, regionprops

# Selects the most left mask in the image
class LeftImageSelector:
    CATEGORY = "FeedFlick"

    @classmethod    
    def INPUT_TYPES(cls):
        return { 
            "required": { 
                "data": ("MASK",),  # Input type is a MASK
            } 
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "choose_image"

    def choose_image(self, data):
        """
        data: A PyTorch tensor with shape [1, H, W].
              Non-zero values represent the mask.
        """
        mask = data  # Input tensor with shape [1, H, W]

        # Print the shape for debugging
        print("Input mask shape:", mask.shape)

        # Ensure the input shape is [1, H, W]
        if mask.dim() == 3 and mask.shape[0] == 1:
            # Convert mask to numpy array
            mask_np = mask.cpu().numpy()  # Shape: [1, H, W]

            # Remove the first dimension for processing
            mask_np_squeezed = mask_np[0]  # Shape: [H, W]

            # Label connected components
            labels = label(mask_np_squeezed)

            # Get properties of each region
            regions = regionprops(labels)

            if regions:
                # Find the region with the smallest min x-coordinate of its pixels
                leftmost_region = min(
                    regions, key=lambda r: r.coords[:, 1].min()
                )  # r.coords[:, 1] are the column indices

                # Get the label of the leftmost region
                leftmost_label = leftmost_region.label

                # Create a new mask with only the leftmost region
                result_np = (labels == leftmost_label).astype(mask_np.dtype)

                # Add back the first dimension
                result_np = result_np[np.newaxis, :, :]

                # Convert back to torch tensor
                result = torch.from_numpy(result_np).to(mask.device)
            else:
                # No regions found, return zero mask
                result = torch.zeros_like(mask)
        else:
            raise ValueError(f"Unsupported mask shape {mask.shape}. Expected [1, H, W].")

        return (result,)



# Selects the most left mask in the image
class RightImageSelector:
    CATEGORY = "FeedFlick"

    @classmethod    
    def INPUT_TYPES(cls):
        return { 
            "required": { 
                "data": ("MASK",),  # Input type is a MASK
            } 
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "choose_image"

    def choose_image(self, data):
        """
        data: A PyTorch tensor with shape [1, H, W].
              Non-zero values represent the mask.
        """
        mask = data  # Input tensor with shape [1, H, W]

        # Print the shape for debugging
        print("Input mask shape:", mask.shape)

        # Ensure the input shape is [1, H, W]
        if mask.dim() == 3 and mask.shape[0] == 1:
            # Convert mask to numpy array
            mask_np = mask.cpu().numpy()  # Shape: [1, H, W]

            # Remove the first dimension for processing
            mask_np_squeezed = mask_np[0]  # Shape: [H, W]

            # Label connected components
            labels = label(mask_np_squeezed)

            # Get properties of each region
            regions = regionprops(labels)

            if regions:
                # Find the region with the smallest min x-coordinate of its pixels
                leftmost_region = max(
                    regions, key=lambda r: r.coords[:, 1].min()
                )  # r.coords[:, 1] are the column indices

                # Get the label of the leftmost region
                leftmost_label = leftmost_region.label

                # Create a new mask with only the leftmost region
                result_np = (labels == leftmost_label).astype(mask_np.dtype)

                # Add back the first dimension
                result_np = result_np[np.newaxis, :, :]

                # Convert back to torch tensor
                result = torch.from_numpy(result_np).to(mask.device)
            else:
                # No regions found, return zero mask
                result = torch.zeros_like(mask)
        else:
            raise ValueError(f"Unsupported mask shape {mask.shape}. Expected [1, H, W].")

        return (result,)

# Fills one image with the color from a specific pixel of another image
class FillWithColor:
    CATEGORY = "FeedFlick"

    @classmethod    
    def INPUT_TYPES(cls):
        return { 
            "required": { 
                "source_image": ("IMAGE",),  # The image to sample the color from
                "target_image": ("IMAGE",),  # The image to fill
                "row": ("INT", {"default": 0}),  # Row index of the pixel to sample
                "column": ("INT", {"default": 0}),  # Column index of the pixel to sample
            } 
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "fill_image"

    def fill_image(self, source_image, target_image, row, column):
        """
        source_image: A PyTorch tensor with shape [1, C, H1, W1].
        target_image: A PyTorch tensor with shape [1, C, H2, W2].
        row, column: Pixel coordinates to sample from the source image.
        """
        # Ensure the input shapes are [1, C, H, W]
        if source_image.dim() != 4 or target_image.dim() != 4:
            raise ValueError("Both input images must have shape [1, C, H, W].")
        
        _, C, H_source, W_source = source_image.shape
        _, _, H_target, W_target = target_image.shape

        # Validate row and column indices
        if row >= H_source or column >= W_source:
            raise ValueError(f"Requested pixel ({row}, {column}) is out of bounds for source image with shape {H_source}x{W_source}.")

        # Sample the pixel at (row, column)
        color = source_image[:, :, row, column]  # Shape: [1, C]

        # Expand the sampled color to match the target image dimensions
        filled_image = color.unsqueeze(-1).unsqueeze(-1).expand(1, C, H_target, W_target)

        return (filled_image,)


import random
import json
import os

class DatasetCreator:
    CATEGORY = "FeedFlick"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_images": ("INT", {"default": 100, "min": 1, "max": 10000}),  # Number of images to generate
                "output_folder": ("STRING", {"default": "dataset_output"}),       # Output folder
                "start_index": ("INT", {"default": 0, "min": 0}),                # Starting index for filenames
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")  # Returns the captions file path, last prompt, and final index
    FUNCTION = "generate_dataset"

    # Initialize default values for styles, roles, and traits
    styles = [
       
    # Traditional Art Styles
    "low poly", "pixel art", "hand-drawn", "watercolor", "oil painting",
    "charcoal sketch", "comic book", "manga", "anime", "cartoon",
    "ink wash", "crayon drawing", "pastel tones", "woodcut print",
    "vintage etching", "monochrome sketch", "crosshatch shading",

    # Digital and Modern Art Styles
    "2D flat art", "3D rendering", "photo-realistic", "hyperrealistic",
    "vector art", "isometric", "cel-shaded", "digital painting",
    "stylized rendering", "VR-inspired", "holographic art", 
    "futuristic digital", "glitch art", "neon-lit", "chrome-finish",

    # Historical and Cultural Art Styles
    "art nouveau", "art deco", "renaissance", "baroque", 
    "romanticism", "gothic illustration", "medieval manuscript",
    "ancient fresco", "egyptian hieroglyphic style", "chinese ink painting",
    "japanese ukiyo-e", "persian miniature", "mayan glyph art",

    # Experimental and Unique Styles
    "abstract", "surreal", "cubist", "pop art", "expressionist",
    "impressionist", "vaporwave", "dreamlike", "psychedelic",
    "holographic overlay", "chrome and neon", "fractals", 
    "multi-textured collage", "bioluminescent theme",
    "negative space art", "parallax effect",

    # Thematic Visual Styles
    "storybook illustration", "children's book art", "dark fantasy painting",
    "celestial map style", "steampunk diagram", "retro sci-fi pulp",
    "post-apocalyptic grunge", "tribal illustration", "organic pattern art",
    "mechanical blueprint", "space-themed digital art", "cosmic horror visuals",
    "foggy atmospheric rendering", "glowing rune-covered imagery",
    "floating ink blots", "stained glass art", "magic glyph designs"
 
    ]

    humanoid_roles = [
        "vampire", "scout", "nurse", "wizard", "robot", "soldier",
        "knight", "alien", "pirate", "ninja", "princess",
        "mechanic", "scientist", "demon", "angel", "superhero",
        "hunter", "ghost", "samurai", "zombie", "cyborg",
        "witch", "orc", "goblin", "elf", "dwarf", "giant",
        "time traveler", "astronaut", "space explorer", "fairy",
        "succubus", "angelic warrior", "ancient deity", "village chief",
        "merchant", "assassin", "thief", "bard", "blacksmith",
        "healer", "shaman", "sorcerer", "aristocrat", "king",
        "queen", "peasant", "monk", "priest", "warlock",
        "jester", "druid", "alchemist", "gladiator", "mercenary",
        "paladin", "templar", "hunter-gatherer", "cybernetic warrior",
        "android", "urban detective", "spy", "high priestess", "cultist",
        "explorer", "herbalist", "necromancer", "clergy member", "pilgrim",
        "villager", "noble", "outlaw", "swordsman", "magician",
        "enchanter", "psychic", "bounty hunter", "ranger"
    ]

    character_roles = [
        "monkey", "dragon", "wolf", "tiger", "cat",
        "phoenix", "unicorn", "shadowy figure", "tree-like creature",
        "fox", "snake", "lion", "spider", "golem",
        "skeleton warrior", "alien creature", "sea serpent",
        "griffin", "hippogriff", "kraken", "wyvern", "hydra",
        "cerberus", "basilisk", "cockatrice", "manticore",
        "slime creature", "tentacled horror", "cloud being",
        "lava monster", "rock elemental", "water elemental",
        "fire elemental", "ice elemental", "light elemental",
        "dark elemental", "sentient orb", "living statue",
        "sapling guardian", "ghostly apparition", "zombie horde",
        "swarm of insects", "flock of birds", "pack of wolves",
        "dinosaur", "raptor", "saber-toothed tiger", "dire wolf",
        "giant snake", "cosmic entity", "eldritch horror",
        "bio-mechanical beast", "cybernetic drone", "energy being",
        "crystal creature", "floating eye", "starfish alien",
        "octopus", "jellyfish", "fungal monster", "mushroom creature",
        "swamp monster", "desert beast", "arctic beast", "polar bear",
        "walrus", "mammoth", "bird of prey", "pegasus", "feral beast"
    ]
    
    character_roles.extend(humanoid_roles)

    genders = ["male", "female", "non-binary", "genderless"]
    outfits = [
    # General Clothing
    "armor", "robe", "suit", "casual clothes", "traditional clothing",
    "high-tech suit", "battle gear", "ceremonial outfit", "mystical robes", 

    # Fantasy and Historical Outfits
    "leather armor", "chainmail", "plate armor", "wizard's robe", 
    "assassin's cloak", "druidic garb", "viking attire", 
    "medieval tunic", "royal robes", "elven armor", 
    "orcish battle gear", "tribal garb", "bard's outfit", 
    "witch's hat and cloak", "ancient warrior's attire", "samurai armor", 
    "shaman's headdress", "gladiator's gear", "pirate captain's coat",

    # Futuristic and Sci-Fi Outfits
    "spacesuit", "exosuit", "cybernetic armor", "nano-fiber suit",
    "holographic attire", "stealth suit", "mech pilot gear",
    "energy shield armor", "battle mech suit", "power armor",
    "alien ceremonial dress", "cosmic explorer's suit", "plasma-infused armor",
    "anti-gravity outfit", "neon-lit jacket", "bio-mechanical suit",

    # Cultural and Traditional Clothing
    "kimono", "sari", "hanbok", "kilt", "tunic and sandals",
    "feathered headdress", "ceremonial tribal outfit", "traditional African robes",
    "Inuit parka", "Native American regalia", "Mexican charro suit",
    "Chinese qipao", "Roman toga", "Greek chiton", "Indian sherwani",
    "Middle Eastern thawb", "Celtic warrior attire",

    # Modern and Urban Styles
    "leather jacket", "denim overalls", "hoodie and jeans",
    "business suit", "streetwear outfit", "punk leather ensemble",
    "military uniform", "sports jersey", "biker gear",
    "formal evening gown", "hipster attire", "workout gear",
    "tracksuit", "combat boots and fatigues", "sneakers and a hoodie",

    # Thematic and Unique Outfits
    "steampunk attire", "gothic dress", "victorian suit",
    "retro 80s jumpsuit", "cyberpunk jacket", "post-apocalyptic rags",
    "vaporwave outfit", "glitch-themed attire", "dreamlike flowing dress",
    "glowing rune-covered robes", "floral-patterned dress",
    "scaly dragon-hide armor", "feathered ceremonial outfit",
    "insect-themed chitin armor", "organic plant-based attire",
    "crystal-embedded armor", "ice-encrusted suit",

    # Mystical and Magical Outfits
    "enchanted robes",
    "robes covered in glowing sigils", "shadowy assassin's garb",
    "sparkling fairy dress", "witch's ceremonial dress",
    "celestial robes with stars", "demonic battle armor",
    ]

    special_traits = [
      
    ]

    def __init__(self):
        # Persistent index stored in an internal state
        self.index = 0

    def set_index(self, start_index):
        """
        Sets the starting index for the generator.
        """
        self.index = start_index

    def generate_prompt(self):
        """
        Generates a random prompt and corresponding caption.
        """
        style = random.choice(self.styles)
        character_role = random.choice(self.character_roles)
        background = "a white background"  # Background is fixed to white

        if character_role in self.humanoid_roles:
            gender = random.choice(self.genders)
            outfit = random.choice(self.outfits)
            if random.random() < 0.5:
                outfit_description = ""
            else:
                outfit_description = f"wearing {outfit}"

            prompt = (
                f"A {character_role}, {gender}, {outfit_description}, "
                f"shown in a side-by-side composition: the left side shows the character from the front, "
                f"and the right side shows the character from the back, both on {background}. The Image is in a {style} style"
            )
            caption = (
                f"A {character_role}, {gender}, {outfit_description} on {background}. The Image is in a {style} style"
            )
        else:
            prompt = (
                f"A {character_role}, shown in a side-by-side composition: "
                f"the left side shows the character from the front, and the right side shows the character from the back, both on {background}. The Image is in a {style} style"
            )
            caption = (
                f"A {character_role}, on {background}. The Image is in a {style} style"
            )

        return prompt, caption

    def generate_dataset(self, num_images, output_folder, start_index=0):
        """
        Generates a dataset with captions and prompts. The index increments persistently.
        """
        os.makedirs(output_folder, exist_ok=True)
        self.set_index(start_index)  # Initialize index from the start_index
        dataset = {}

        for _ in range(num_images):
            prompt, caption = self.generate_prompt()
            image_filename = f"image_{self.index:03d}.jpg"  # Format index into filename
            dataset[image_filename] = caption
            self.index += 1  # Increment index after each entry

        # Save captions to a JSON file
        captions_file = os.path.join(output_folder, "captions.json")
        with open(captions_file, "w") as f:
            json.dump(dataset, f, indent=4)

        return (captions_file, prompt, self.index)


class Number_Counter:
    def __init__(self):
        self.counters = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "number_type": (["integer", "float"],),
                "mode": (["increment", "decrement", "increment_to_stop", "decrement_to_stop"],),
                "start": ("FLOAT", {"default": 0, "min": -18446744073709551615, "max": 18446744073709551615, "step": 0.01}),
                "stop": ("FLOAT", {"default": 0, "min": -18446744073709551615, "max": 18446744073709551615, "step": 0.01}),
                "step": ("FLOAT", {"default": 1, "min": 0, "max": 99999, "step": 0.01}),
            },
            "optional": {
                "reset_bool": ("NUMBER",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    RETURN_TYPES = ("NUMBER", "FLOAT", "INT")
    RETURN_NAMES = ("number", "float", "int")
    FUNCTION = "increment_number"

    CATEGORY = "WAS Suite/Number"

    def increment_number(self, number_type, mode, start, stop, step, unique_id, reset_bool=0):

        counter = int(start) if mode == 'integer' else start
        if self.counters.__contains__(unique_id):
            counter = self.counters[unique_id]

        if round(reset_bool) >= 1:
            counter = start

        if mode == 'increment':
            counter += step
        elif mode == 'deccrement':
            counter -= step
        elif mode == 'increment_to_stop':
            if counter + step > stop:
                raise ValueError(f"Counter exceeds the stop limit ({stop}) on increment to stop.")
            counter = counter + step if counter < stop else counter
        elif mode == 'decrement_to_stop':
            if counter - step < stop:
                raise ValueError(f"Counter exceeds the stop limit ({stop}) on decrement to stop.")
            counter = counter - step if counter > stop else counter

        self.counters[unique_id] = counter

        result = int(counter) if number_type == 'integer' else float(counter)

        return ( result, float(counter), int(counter) )

