import easyocr
import re

PLATE_REGEX = re.compile(
    r'^(?:AP|AR|AS|BR|CG|CH|DD|DL|DN|GA|GJ|HP|HR|JH|JK|KA|KL|LD|MH|ML|MN|MP|MZ|NL|OD|PB|PY|RJ|SK|TN|TR|TS|UK|UP|WB|AN)\d{1,2}[A-Z]{1,3}\d{4}$'
)

UNWANTED = {"IND", "INDIA", "BHARAT", "GOVT", "IN"}

class PlateOCR:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])

    def read(self, image):
        results = self.reader.readtext(image)
        texts = [r[1] for r in results if r[1].upper() not in UNWANTED]

        text = ''.join(texts).upper().replace(" ", "")
        return text if PLATE_REGEX.match(text) else None
