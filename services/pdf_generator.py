"""
ToonTales AI - PDF Storybook Export Service
Generates professional, publication-quality illustrated cartoon storybooks using ReportLab.
Includes cover page, character profiles, scene illustrations, narrative typography,
and page numbering.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from PIL import Image as PILImage

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak, Table, TableStyle
)
from reportlab.pdfgen import canvas

logger = logging.getLogger("ToonTalesAI.PDFGenerator")


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to calculate total page count and add running header/footer.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_decorations(self, total_pages: int):
        # Exclude decorations on cover page (page 1)
        if self._pageNumber > 1:
            self.saveState()
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#64748b"))
            
            # Running header
            self.drawString(54, 750, "ToonTales AI — Illustrated Digital Storybook")
            self.setStrokeColor(colors.HexColor("#e2e8f0"))
            self.setLineWidth(0.5)
            self.line(54, 742, self._pagesize[0] - 54, 742)

            # Running footer
            self.line(54, 45, self._pagesize[0] - 54, 45)
            self.drawString(54, 32, "Generated with Generative AI")
            self.drawRightString(
                self._pagesize[0] - 54, 32, f"Page {self._pageNumber} of {total_pages}"
            )
            self.restoreState()


class PDFStorybookGenerator:
    def __init__(self):
        self.page_width, self.page_height = letter
        self.margin = 54  # 0.75 in
        self.content_width = self.page_width - (2 * self.margin)

    def generate_pdf(self, story_data: Dict[str, Any], output_path: str) -> str:
        """
        Builds and saves the complete illustrated PDF storybook.
        """
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        doc = SimpleDocTemplate(
            str(dest),
            pagesize=letter,
            leftMargin=self.margin,
            rightMargin=self.margin,
            topMargin=self.margin,
            bottomMargin=self.margin
        )

        styles = getSampleStyleSheet()
        self._setup_custom_styles(styles)

        story_elements = []

        # 1. Build Cover Page
        self._build_cover_page(story_data, styles, story_elements)
        story_elements.append(PageBreak())

        # 2. Build Characters Page
        self._build_characters_page(story_data, styles, story_elements)
        story_elements.append(PageBreak())

        # 3. Build Scene Pages
        self._build_scenes_pages(story_data, styles, story_elements)

        # Build Document
        doc.build(story_elements, canvasmaker=NumberedCanvas)
        logger.info(f"Illustrated PDF storybook created successfully at: {dest.name}")
        return str(dest)

    @staticmethod
    def get_missing_illustrations(story_data: Any) -> List[int]:
        """
        Returns a list of scene numbers that lack a valid image on disk.
        Does not trigger any API calls.
        Safely handles None, non-dict story_data, non-dict scenes, or missing keys.
        """
        if not isinstance(story_data, dict):
            return []
        missing: List[int] = []
        for idx, scene in enumerate(story_data.get("scenes", []) or []):
            if not isinstance(scene, dict):
                if isinstance(scene, int):
                    missing.append(scene)
                continue
            s_num = scene.get("scene_number")
            if s_num is None or not isinstance(s_num, (int, str)):
                s_num = idx + 1
            else:
                try:
                    s_num = int(s_num)
                except (ValueError, TypeError):
                    s_num = idx + 1
            img_path = scene.get("image_path")
            if not img_path or not isinstance(img_path, str) or not os.path.isfile(img_path) or os.path.getsize(img_path) == 0:
                missing.append(s_num)
        return missing

    @staticmethod
    def format_missing_scenes(missing_scenes: Any) -> str:
        """
        Formats a collection of missing scenes into a clean comma-separated string of scene numbers.
        Robustly supports:
        - List of integers: [3, 4, 5] -> "3, 4, 5"
        - List of scene dictionaries: [{"scene_number": 3}, ...] -> "3, ..."
        - Dictionaries with 'scene_id' -> "3, ..."
        - Missing or None scene numbers -> "?"
        - Empty lists, None, or unexpected types -> ""
        """
        if not missing_scenes or not isinstance(missing_scenes, (list, tuple, set)):
            return ""

        formatted: List[str] = []
        for item in missing_scenes:
            if isinstance(item, int):
                formatted.append(str(item))
            elif isinstance(item, str):
                s = item.strip()
                formatted.append(s if s else "?")
            elif isinstance(item, dict):
                val = item.get("scene_number")
                if val is None:
                    val = item.get("scene_id")
                formatted.append(str(val) if val is not None and str(val).strip() else "?")
            else:
                formatted.append("?")
        return ", ".join(formatted)

    def _setup_custom_styles(self, styles):
        """
        Configures typography and color styles tailored for a children's storybook.
        """
        styles.add(ParagraphStyle(
            name="BookCoverTitle",
            fontName="Helvetica-Bold",
            fontSize=28,
            leading=34,
            textColor=colors.HexColor("#1e1b4b"),
            alignment=1,  # Center
            spaceAfter=15
        ))

        styles.add(ParagraphStyle(
            name="BookCoverSubtitle",
            fontName="Helvetica-Oblique",
            fontSize=13,
            leading=18,
            textColor=colors.HexColor("#4338ca"),
            alignment=1,
            spaceAfter=25
        ))

        styles.add(ParagraphStyle(
            name="BookCoverSummary",
            fontName="Helvetica",
            fontSize=11,
            leading=16,
            textColor=colors.HexColor("#334155"),
            alignment=1,
            spaceAfter=30
        ))

        styles.add(ParagraphStyle(
            name="SectionHeading",
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=colors.HexColor("#1e1b4b"),
            spaceAfter=15,
            spaceBefore=10
        ))

        styles.add(ParagraphStyle(
            name="SceneTitle",
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#312e81"),
            spaceAfter=10
        ))

        styles.add(ParagraphStyle(
            name="SceneNumberBadge",
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=12,
            textColor=colors.HexColor("#4f46e5"),
            spaceAfter=4
        ))

        styles.add(ParagraphStyle(
            name="SceneNarration",
            fontName="Helvetica",
            fontSize=11.5,
            leading=17,
            textColor=colors.HexColor("#1e293b"),
            spaceBefore=10,
            spaceAfter=12
        ))

        styles.add(ParagraphStyle(
            name="SceneEnvironment",
            fontName="Helvetica-Oblique",
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#64748b"),
            spaceAfter=6
        ))

        styles.add(ParagraphStyle(
            name="CharName",
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#1e1b4b"),
            spaceAfter=3
        ))

        styles.add(ParagraphStyle(
            name="CharDetail",
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#334155"),
            spaceAfter=8
        ))

    def _build_cover_page(self, story: Dict[str, Any], styles, elements):
        """
        Creates an attractive storybook cover with badge ribbons and metadata.
        """
        elements.append(Spacer(1, 40))

        # Brand header badge
        badge_style = ParagraphStyle(
            name="BrandBadge",
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#4f46e5"),
            alignment=1,
            spaceAfter=15
        )
        elements.append(Paragraph("★ TOONTALES AI STORYBOOK ★", badge_style))
        elements.append(Spacer(1, 20))

        # Title
        title_text = story.get("title", "A Wonderful Tale")
        elements.append(Paragraph(title_text, styles["BookCoverTitle"]))

        # Art style & Genre tag
        genre = story.get("genre", "Adventure")
        art_style = story.get("art_style", "Colorful 2D cartoon")
        meta_line = f"Genre: {genre} &nbsp;|&nbsp; Art Style: {art_style}"
        elements.append(Paragraph(meta_line, styles["BookCoverSubtitle"]))

        elements.append(Spacer(1, 20))

        # Summary box
        summary = story.get("summary", "An exciting animated story created with generative artificial intelligence.")
        summary_html = f"<i>&ldquo;{summary}&rdquo;</i>"
        
        # Wrapped in a decorative table frame
        summary_table = Table(
            [[Paragraph(summary_html, styles["BookCoverSummary"])]],
            colWidths=[self.content_width - 40]
        )
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
            ('ROUNDEDCORNERS', [8, 8, 8, 8]),
            ('TOPPADDING', (0, 0), (-1, -1), 14),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 14),
            ('LEFTPADDING', (0, 0), (-1, -1), 16),
            ('RIGHTPADDING', (0, 0), (-1, -1), 16),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        elements.append(summary_table)

        elements.append(Spacer(1, 40))

        # Academic credit / metadata
        credit_style = ParagraphStyle(
            name="CoverCredit",
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            textColor=colors.HexColor("#64748b"),
            alignment=1
        )
        elements.append(Paragraph("Generated by ToonTales AI &bull; MCA Generative AI Project", credit_style))

    def _build_characters_page(self, story: Dict[str, Any], styles, elements):
        """
        Creates a dedicated character introduction roster.
        """
        elements.append(Spacer(1, 15))
        elements.append(Paragraph("Meet the Characters", styles["SectionHeading"]))
        elements.append(Spacer(1, 10))

        characters = story.get("characters", [])
        if not characters:
            elements.append(Paragraph("No character profiles specified.", styles["CharDetail"]))
            return

        for char in characters:
            name = char.get("name", "Character")
            role = char.get("role", "Character")
            appearance = char.get("appearance", "A colorful cartoon character.")
            personality = char.get("personality", "Friendly and brave.")

            char_text = (
                f"<b>{name}</b> ({role})<br/>"
                f"<b>Visual Appearance:</b> {appearance}<br/>"
                f"<b>Personality:</b> {personality}"
            )

            card_table = Table(
                [[Paragraph(char_text, styles["CharDetail"])]],
                colWidths=[self.content_width]
            )
            card_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#e2e8f0")),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('LEFTPADDING', (0, 0), (-1, -1), 12),
                ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ]))
            elements.append(card_table)
            elements.append(Spacer(1, 12))

    def _build_scenes_pages(self, story: Dict[str, Any], styles, elements):
        """
        Adds each scene on its own page with illustration, heading, and narration.
        """
        scenes = story.get("scenes", [])
        total_scenes = len(scenes)

        for idx, scene in enumerate(scenes):
            scene_num = scene.get("scene_number", idx + 1)
            heading = scene.get("heading", f"Scene {scene_num}")
            narration = scene.get("narration", "")
            environment = scene.get("environment", "")
            image_path = scene.get("image_path")

            elements.append(Spacer(1, 10))

            # Scene counter badge & heading
            elements.append(
                Paragraph(f"SCENE {scene_num} OF {total_scenes}", styles["SceneNumberBadge"])
            )
            elements.append(Paragraph(heading, styles["SceneTitle"]))

            # Scene illustration
            max_img_w = self.content_width
            max_img_h = 300  # Leave room for narration and margins

            if image_path and os.path.isfile(image_path):
                try:
                    # Calculate scaling keeping aspect ratio
                    with PILImage.open(image_path) as pil_img:
                        orig_w, orig_h = pil_img.size
                        ratio = min(max_img_w / orig_w, max_img_h / orig_h)
                        target_w = orig_w * ratio
                        target_h = orig_h * ratio

                    rl_img = RLImage(image_path, width=target_w, height=target_h)
                    rl_img.hAlign = 'CENTER'
                    elements.append(rl_img)
                    elements.append(Spacer(1, 12))
                except Exception as img_err:
                    logger.warning(f"Failed to embed image {image_path}: {img_err}")
                    elements.append(self._create_image_placeholder(scene_num, styles))
            else:
                elements.append(self._create_image_placeholder(scene_num, styles))

            # Narration
            elements.append(Paragraph(narration, styles["SceneNarration"]))

            # Environment footnote
            if environment:
                elements.append(
                    Paragraph(f"<b>Setting:</b> {environment}", styles["SceneEnvironment"])
                )

            # Page break after every scene except the last one
            if idx < total_scenes - 1:
                elements.append(PageBreak())

    def _create_image_placeholder(self, scene_num: int, styles):
        """
        Graceful visual placeholder card if an image is missing or failed to generate.
        """
        placeholder_text = (
            f"<i>[ Scene {scene_num} Illustration pending or not generated ]</i>"
        )
        table = Table(
            [[Paragraph(placeholder_text, styles["BookCoverSummary"])]],
            colWidths=[self.content_width],
            rowHeights=[140]
        )
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        return table
