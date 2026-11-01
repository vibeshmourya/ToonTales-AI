"""
Script to create a demonstration storybook in generated_stories/demo_story
Demonstrates end-to-end image generation, PDF publication, and story persistence.
"""

import os
import sys

# Ensure root workspace is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from services.image_generator import ImageGeneratorService
from services.pdf_generator import PDFStorybookGenerator
from services.narration import NarrationService
from utils.helpers import save_story_json

def create_demo():
    story = {
        "title": "Ruby and the Moonbeam Lantern",
        "summary": "Ruby the little white rabbit dreams of touching the moon and embarks on a whimsical flight with her star-badge suit.",
        "genre": "Adventure",
        "art_style": "Colorful 2D cartoon",
        "characters": [{
            "name": "Ruby",
            "role": "Curious Adventurer",
            "appearance": "Small white rabbit with long velvety ears, purple astronaut suit, silver helmet, and a bright star badge.",
            "personality": "Brave, cheerful, and full of wonder."
        }],
        "scenes": [
            {
                "scene_number": 1,
                "heading": "Stargazing from the Meadow",
                "narration": "Every night, Ruby sat in the clover meadow, gazing up at the smiling crescent moon through her brass telescope.",
                "image_prompt": "A cute small white rabbit in a purple astronaut suit looking through a telescope at the moon in a whimsical night meadow.",
                "characters_present": ["Ruby"],
                "environment": "Enchanted clover field under a navy night sky with gentle fireflies"
            },
            {
                "scene_number": 2,
                "heading": "Building the Carrot Rocket",
                "narration": "With wooden planks, shiny tin foil, and lots of giggles, Ruby constructed her cozy little spaceship.",
                "image_prompt": "Ruby the white rabbit in her purple astronaut suit tightening bolts on a friendly round orange rocket ship.",
                "characters_present": ["Ruby"],
                "environment": "Cozy treehouse workshop illuminated by fairy lights"
            },
            {
                "scene_number": 3,
                "heading": "Blastoff to the Stars!",
                "narration": "Three, two, one, blastoff! Ruby rocketed into the twilight sky surrounded by sparkling constellation dust.",
                "image_prompt": "A friendly orange rocket soaring through colorful pastel nebulae with Ruby waving from the round porthole.",
                "characters_present": ["Ruby"],
                "environment": "Whimsical deep space glowing with pastel purples and golden stars"
            }
        ]
    }

    demo_dir = os.path.join("generated_stories", "demo_story")
    os.makedirs(demo_dir, exist_ok=True)
    
    img_svc = ImageGeneratorService(provider="pollinations")
    tts_svc = NarrationService()

    print("Generating cartoon scene illustrations...")
    for sc in story["scenes"]:
        s_num = sc["scene_number"]
        img_path = os.path.join(demo_dir, f"scene_{s_num:02d}.png")
        print(f"Generating image for scene {s_num}...")
        img_svc.generate_single_scene_image(sc, story["characters"], story["art_style"], img_path)
        sc["image_path"] = img_path
        
        # Audio narration
        audio_path = os.path.join(demo_dir, f"scene_{s_num:02d}.mp3")
        tts_svc.generate_narration(sc["narration"], audio_path)

    print("Generating PDF storybook...")
    pdf_svc = PDFStorybookGenerator()
    pdf_path = os.path.join(demo_dir, "story.pdf")
    pdf_svc.generate_pdf(story, pdf_path)
    
    json_path = os.path.join(demo_dir, "story.json")
    save_story_json(story, json_path)
    print("Demo creation complete! Output saved in generated_stories/demo_story")

if __name__ == "__main__":
    create_demo()
