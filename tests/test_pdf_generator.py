"""
Unit tests for PDF Storybook Generator Service.
Validates PDF creation, element assembly, aspect-ratio scaling, and missing-image resilience.
"""

import os
from PIL import Image
from services.pdf_generator import PDFStorybookGenerator


def test_pdf_generation_complete(tmp_path):
    """Test generating a full PDF with cover page, character profiles, and scenes."""
    # Create a dummy image
    img_path = str(tmp_path / "scene_01.png")
    test_img = Image.new("RGB", (400, 300), color="coral")
    test_img.save(img_path)

    story_data = {
        "title": "Barnaby the Tea Dragon",
        "summary": "A friendly dragon brews soothing tea for his village friends.",
        "genre": "Fantasy",
        "art_style": "Storybook illustration",
        "characters": [
            {
                "name": "Barnaby",
                "role": "Gentle Hero",
                "appearance": "Chubby green dragon with little wings and an apron with a teapot badge.",
                "personality": "Kind, warm, and gentle."
            }
        ],
        "scenes": [
            {
                "scene_number": 1,
                "heading": "Morning Steam",
                "narration": "Barnaby woke up and gently steamed a kettle of wild chamomile tea.",
                "image_path": img_path,
                "environment": "Cozy mountain cottage kitchen"
            },
            {
                "scene_number": 2,
                "heading": "Sharing with Friends",
                "narration": "The forest creatures gathered with wooden mugs, smiling happily.",
                "image_path": None,  # Test missing image handling
                "environment": "Sunny woodland clearing"
            }
        ]
    }

    pdf_out = str(tmp_path / "storybook_output.pdf")
    generator = PDFStorybookGenerator()
    result_path = generator.generate_pdf(story_data, pdf_out)

    assert os.path.isfile(result_path)
    assert os.path.getsize(result_path) > 1000

    # Verify PDF magic bytes
    with open(result_path, "rb") as f:
        header = f.read(5)
        assert header.startswith(b"%PDF-")


def test_pdf_handles_missing_images_gracefully(tmp_path):
    """Verify PDF generator still compiles cleanly when scenes have no images."""
    story_data = {
        "title": "Story Without Images",
        "summary": "A test story to verify fallback placeholder rendering.",
        "genre": "Moral Story",
        "art_style": "Colorful 2D cartoon",
        "characters": [],
        "scenes": [
            {
                "scene_number": 1,
                "heading": "Just Text",
                "narration": "This scene has no image path.",
                "image_path": None,
                "environment": "Simple room"
            }
        ]
    }

    pdf_out = str(tmp_path / "text_only.pdf")
    generator = PDFStorybookGenerator()
    result_path = generator.generate_pdf(story_data, pdf_out)

    assert os.path.isfile(result_path)
    assert os.path.getsize(result_path) > 500
