import unittest
from PIL import Image, ImageDraw
import pandas as pd

from ocr_utils import enhance_image, text_to_dataframe, build_quality_report, build_excel


class OCRUtilityTests(unittest.TestCase):
    def test_empty_text_returns_empty_frame(self):
        result = text_to_dataframe("")
        self.assertEqual(list(result.columns), ["Extracted text"])
        self.assertEqual(len(result), 0)

    def test_key_value_lines_are_structured(self):
        result = text_to_dataframe("Name: Ali\nAge: 22\nCity: Karachi")
        self.assertEqual(list(result.columns), ["Field", "Value"])
        self.assertEqual(result.iloc[0]["Value"], "Ali")
        self.assertEqual(len(result), 3)

    def test_pipe_table_is_structured(self):
        result = text_to_dataframe("Name | Age\nAli | 22")
        self.assertEqual(list(result.columns), ["Column 1", "Column 2"])
        self.assertEqual(result.iloc[1, 0], "Ali")

    def test_plain_lines_are_preserved(self):
        result = text_to_dataframe("First line\nSecond line")
        self.assertEqual(list(result.columns), ["Extracted text"])
        self.assertEqual(len(result), 2)

    def test_enhancement_returns_pil_image(self):
        image = Image.new("RGB", (400, 200), "white")
        draw = ImageDraw.Draw(image)
        draw.text((20, 50), "Test 123", fill="black")
        enhanced = enhance_image(image)
        self.assertIsInstance(enhanced, Image.Image)
        self.assertEqual(enhanced.mode, "L")

    def test_excel_is_valid_xlsx(self):
        df = pd.DataFrame({"Field": ["Name"], "Value": ["Ali"]})
        report = {"Words detected": 2, "Average OCR confidence (%)": 90.0}
        data = build_excel(df, "Name: Ali", report)
        self.assertTrue(data.startswith(b"PK"))
        self.assertGreater(len(data), 500)

    def test_quality_report_has_expected_metrics(self):
        tokens = pd.DataFrame({"text": ["Ali", "22"]})
        df = pd.DataFrame({"Extracted text": ["Ali 22"]})
        report = build_quality_report("test.png", "English", "Ali 22", tokens, df, 88.4, 1.2)
        self.assertEqual(report["Words detected"], 2)
        self.assertEqual(report["Rows extracted"], 1)
        self.assertEqual(report["Average OCR confidence (%)"], 88.4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
