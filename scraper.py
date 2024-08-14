import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin
import concurrent.futures
import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass, field, asdict
import unicodedata
import time
import random
import json
import spacy
from spacy.training import Example
import joblib
import os
from typing import NamedTuple

@dataclass
class CompanyInfo:
    url: str
    company_name: str = "N/A"
    contact_name: str = "N/A"
    email: str = "N/A"
    confidence: dict = field(default_factory=dict)

class PageInfo(NamedTuple):
    url: str
    relevance_score: int

class WebScraper:
    def __init__(self, timeout: int = 10, max_retries: int = 3, max_depth: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_depth = max_depth
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.load_extraction_patterns()
        self.load_nlp_model()

    def load_extraction_patterns(self):
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

    def load_nlp_model(self):
        try:
            self.nlp = spacy.load("de_core_news_sm")
        except IOError:
            print("Downloading German NLP model...")
            spacy.cli.download("de_core_news_sm")
            self.nlp = spacy.load("de_core_news_sm")

    def train_nlp_model(self, training_data: List[Tuple[str, dict]]):
        nlp = spacy.blank("de")

        if "ner" not in nlp.pipe_names:
            ner = nlp.add_pipe("ner", last=True)
        else:
            ner = nlp.get_pipe("ner")

        for _, annotations in training_data:
            for ent in annotations.get("entities"):
                ner.add_label(ent[2])

        optimizer = nlp.begin_training()
        for itn in range(20):
            random.shuffle(training_data)
            losses = {}
            for text, annotations in training_data:
                example = Example.from_dict(nlp.make_doc(text), annotations)
                nlp.update([example], drop=0.5, losses=losses)
            print(f"Iteration {itn}: Losses {losses}")

        self.nlp = nlp

    def fetch_url(self, url: str) -> Optional[str]:
        for _ in range(self.max_retries):
            try:
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                return response.text
            except requests.RequestException as e:
                logging.warning(f"Error fetching {url}: {e}")
                time.sleep(random.uniform(1, 3))
        return None

    def find_impressum_page(self, soup: BeautifulSoup, base_url: str) -> Optional[str]:
        impressum_keywords = ['impressum', 'imprint', 'legal', 'kontakt', 'contact', 'about', 'über uns']
        candidate_links = []

        for link in soup.find_all('a', href=True):
            href = link['href']
            text = link.get_text().lower()
            if any(keyword in text or keyword in href.lower() for keyword in impressum_keywords):
                full_url = urljoin(base_url, href)
                candidate_links.append(full_url)

        return candidate_links[0] if candidate_links else None

    def normalize_text(self, text: str) -> str:
        return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')

    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[\r\n]+', ' ', text)
        return text

    def extract_company_name(self, text: str) -> str:
        doc = self.nlp(text)
        org_names = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        
        patterns = [
            r'([A-Z][a-zäöüß]+(?:\s+[A-Z][a-zäöüß]+){0,2}\s+(?:GmbH|AG|PartG mbB|PartGmbB|GbR|mbH|OHG|KG|e\.V\.))',
            r'Kanzlei\s+([A-Z][a-zäöüß]+(?:\s+[A-Z][a-zäöüß]+)*)',
            r'Rechtsanwaltskanzlei\s+([A-Z][a-zäöüß]+(?:\s+[A-Z][a-zäöüß]+)*)'
        ]
        
        regex_names = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            regex_names.extend(matches)

        all_names = org_names + regex_names
        return all_names[0] if all_names else 'N/A'

    def extract_contact_name(self, text: str) -> str:
        doc = self.nlp(text)
        person_names = [ent.text for ent in doc.ents if ent.label_ == "PER"]

        patterns = [
            r'(Herr|Frau)\s+([A-Z][a-zäöüß]+(?:\s+[A-Z][a-zäöüß]+)*)',
            r'(Dr.|Prof.)\s+([A-Z][a-zäöüß]+(?:\s+[A-Z][a-zäöüß]+)*)'
        ]
        
        regex_names = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            regex_names.extend(matches)

        all_names = person_names + regex_names
        return all_names[0] if all_names else 'N/A'

    def extract_email(self, text: str) -> str:
        emails = self.email_pattern.findall(text)
        return emails[0] if emails else 'N/A'

    def extract_info_from_impressum(self, url: str, html_content: str) -> CompanyInfo:
        soup = BeautifulSoup(html_content, 'html.parser')
        text = self.normalize_text(soup.get_text(separator=" ", strip=True))
        text = self.clean_text(text)

        company_name = self.extract_company_name(text)
        contact_name = self.extract_contact_name(text)
        email = self.extract_email(text)

        info = CompanyInfo(url)
        info.company_name = company_name
        info.contact_name = contact_name
        info.email = email

        info.confidence['company_name'] = 0.9 if company_name != 'N/A' else 0.5
        info.confidence['contact_name'] = 0.9 if contact_name != 'N/A' else 0.5
        info.confidence['email'] = 0.9 if email != 'N/A' else 0.5

        return info

    def extract_info_from_url(self, url: str) -> Optional[CompanyInfo]:
        html_content = self.fetch_url(url)
        if not html_content:
            return None

        soup = BeautifulSoup(html_content, 'html.parser')
        impressum_url = self.find_impressum_page(soup, url)

        if impressum_url:
            impressum_content = self.fetch_url(impressum_url)
            if impressum_content:
                info = self.extract_info_from_impressum(impressum_url, impressum_content)
                if info.company_name != 'N/A' and info.contact_name != 'N/A' and info.email != 'N/A':
                    return info

        return self.extract_info_from_impressum(url, html_content)

def main(urls: List[str], max_workers: int = 5) -> List[CompanyInfo]:
    scraper = WebScraper()

    isolated_training_data = [
       ("Rechtsanwaltskanzlei Schmidt", {"entities": [(0, 26, "ORG")]}), ("Anwaltsbüro Müller & Partner", {"entities": [(0, 27, "ORG")]}), ("Kanzlei Mustermann & Partner Rechtsanwälte", {"entities": [(0, 41, "ORG")]}), ("Rechtsanwälte Fischer & Söhne GmbH", {"entities": [(0, 32, "ORG")]}), ("Anwaltskanzlei Hansen & Partner mbB", {"entities": [(0, 33, "ORG")]}), ("Rechtsanwaltskanzlei Dr. Becker GmbH", {"entities": [(0, 35, "ORG")]}), ("Kanzlei Müller und Partner mbB", {"entities": [(0, 31, "ORG")]}), ("Kanzlei Berger und Partner", {"entities": [(0, 24, "ORG")]}), ("Rechtsanwälte Dr. Maier & Kollegen", {"entities": [(0, 33, "ORG")]}), ("Rechtsanwaltskanzlei Franke", {"entities": [(0, 26, "ORG")]}), ("Kanzlei Schmidt und Partner mbB", {"entities": [(0, 29, "ORG")]}), ("Anwaltskanzlei Hofmann und Partner", {"entities": [(0, 33, "ORG")]}), ("Kanzlei Schneider & Kollegen", {"entities": [(0, 26, "ORG")]}), ("Rechtsanwaltskanzlei Müller GmbH", {"entities": [(0, 31, "ORG")]}), ("Anwaltskanzlei Schulze & Kollegen GbR", {"entities": [(0, 35, "ORG")]}), ("Kanzlei Weber & Partner", {"entities": [(0, 22, "ORG")]}), ("Rechtsanwälte Fischer und Partner", {"entities": [(0, 32, "ORG")]}), ("Kanzlei Braun und Kollegen", {"entities": [(0, 26, "ORG")]}), ("Rechtsanwälte Meier & Söhne AG", {"entities": [(0, 32, "ORG")]}), ("Anwaltskanzlei Schröder mbH", {"entities": [(0, 27, "ORG")]}), ("Kanzlei Krüger & Partner", {"entities": [(0, 22, "ORG")]}), ("Rechtsanwälte Wolf und Kollegen", {"entities": [(0, 32, "ORG")]}), ("Kanzlei Vogel & Partner", {"entities": [(0, 21, "ORG")]}), ("Rechtsanwälte Lutz und Partner", {"entities": [(0, 31, "ORG")]}), ("Kanzlei Beck & Kollegen", {"entities": [(0, 21, "ORG")]}), ("Kanzlei Richter und Partner mbB", {"entities": [(0, 30, "ORG")]}), ("Rechtsanwälte Wagner & Partner", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Jung und Kollegen", {"entities": [(0, 24, "ORG")]}), ("Rechtsanwälte Hoffmann und Partner", {"entities": [(0, 35, "ORG")]}), ("Rechtsanwaltskanzlei Kaiser GmbH", {"entities": [(0, 32, "ORG")]}), ("Anwaltskanzlei Scholz & Partner mbB", {"entities": [(0, 33, "ORG")]}), ("Kanzlei Schmitt und Partner", {"entities": [(0, 27, "ORG")]}), ("Rechtsanwälte Schmid & Kollegen", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Lehmann & Partner", {"entities": [(0, 25, "ORG")]}), ("Rechtsanwälte Schulze und Partner", {"entities": [(0, 33, "ORG")]}), ("Kanzlei Fuchs und Partner mbB", {"entities": [(0, 28, "ORG")]}), ("Anwaltskanzlei Herrmann & Kollegen", {"entities": [(0, 34, "ORG")]}), ("Rechtsanwaltskanzlei Neumann", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Meyer und Partner", {"entities": [(0, 24, "ORG")]}), ("Rechtsanwälte Schmidt & Partner", {"entities": [(0, 30, "ORG")]}), ("Anwaltskanzlei Bauer & Kollegen", {"entities": [(0, 31, "ORG")]}), ("Kanzlei Krüger und Partner", {"entities": [(0, 25, "ORG")]}), ("Rechtsanwälte Maier & Kollegen", {"entities": [(0, 31, "ORG")]}), ("Anwaltskanzlei Braun & Partner", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Richter und Partner mbB", {"entities": [(0, 30, "ORG")]}), ("Rechtsanwälte Wagner & Partner", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Jung und Kollegen", {"entities": [(0, 24, "ORG")]}), ("Rechtsanwälte Hoffmann und Partner", {"entities": [(0, 35, "ORG")]}), ("Rechtsanwaltskanzlei Neumann", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Meyer und Partner", {"entities": [(0, 24, "ORG")]}), ("Rechtsanwälte Schmidt & Partner", {"entities": [(0, 30, "ORG")]}), ("Anwaltskanzlei Bauer & Kollegen", {"entities": [(0, 31, "ORG")]}), ("Kanzlei Krüger und Partner", {"entities": [(0, 25, "ORG")]}), ("Rechtsanwälte Maier & Kollegen", {"entities": [(0, 31, "ORG")]}), ("Anwaltskanzlei Braun & Partner", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Richter und Partner mbB", {"entities": [(0, 30, "ORG")]}), ("Rechtsanwälte Wagner & Partner", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Jung und Kollegen", {"entities": [(0, 24, "ORG")]}), ("Rechtsanwälte Hoffmann und Partner", {"entities": [(0, 35, "ORG")]}), ("Rechtsanwaltskanzlei Neumann", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Meyer und Partner", {"entities": [(0, 24, "ORG")]}), ("Rechtsanwälte Schmidt & Partner", {"entities": [(0, 30, "ORG")]}), ("Anwaltskanzlei Bauer & Kollegen", {"entities": [(0, 31, "ORG")]}), ("Kanzlei Krüger und Partner", {"entities": [(0, 25, "ORG")]}), ("Rechtsanwälte Maier & Kollegen", {"entities": [(0, 31, "ORG")]}), ("Anwaltskanzlei Braun & Partner", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Richter und Partner mbB", {"entities": [(0, 30, "ORG")]}), ("Rechtsanwälte Wagner & Partner", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Jung und Kollegen", {"entities": [(0, 24, "ORG")]}), ("Rechtsanwälte Hoffmann und Partner", {"entities": [(0, 35, "ORG")]}), ("Rechtsanwaltskanzlei Neumann", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Meyer und Partner", {"entities": [(0, 24, "ORG")]}), ("Rechtsanwälte Schmidt & Partner", {"entities": [(0, 30, "ORG")]}), ("Anwaltskanzlei Bauer & Kollegen", {"entities": [(0, 31, "ORG")]}), ("Kanzlei Krüger und Partner", {"entities": [(0, 25, "ORG")]}), ("Rechtsanwälte Maier & Kollegen", {"entities": [(0, 31, "ORG")]}), ("Anwaltskanzlei Braun & Partner", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Richter und Partner mbB", {"entities": [(0, 30, "ORG")]}), ("Rechtsanwälte Wagner & Partner", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Jung und Kollegen", {"entities": [(0, 24, "ORG")]}), ("Rechtsanwälte Hoffmann und Partner", {"entities": [(0, 35, "ORG")]}), ("Rechtsanwaltskanzlei Neumann", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Meyer und Partner", {"entities": [(0, 24, "ORG")]}), ("Rechtsanwälte Schmidt & Partner", {"entities": [(0, 30, "ORG")]}), ("Anwaltskanzlei Bauer & Kollegen", {"entities": [(0, 31, "ORG")]}), ("Kanzlei Krüger und Partner", {"entities": [(0, 25, "ORG")]}), ("Rechtsanwälte Maier & Kollegen", {"entities": [(0, 31, "ORG")]}), ("Anwaltskanzlei Braun & Partner", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Richter und Partner mbB", {"entities": [(0, 30, "ORG")]}), ("Rechtsanwälte Wagner & Partner", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Jung und Kollegen", {"entities": [(0, 24, "ORG")]}),
("Dr. Hans Mustermann", {"entities": [(0, 18, "PER")]}), ("Erika Musterfrau", {"entities": [(0, 16, "PER")]}), ("Max Mustermann", {"entities": [(0, 14, "PER")]}), ("Rechtsanwalt Dr. Hans Mustermann", {"entities": [(14, 32, "PER")]}), ("RA Erika Musterfrau", {"entities": [(3, 18, "PER")]}), ("Fachanwalt für Familienrecht Klaus Weber", {"entities": [(30, 41, "PER")]}), ("Kontakt: Dr. Andrea Schulze, LL.M.", {"entities": [(9, 31, "PER")]}), ("Anwalt Dr. Michael Bauer", {"entities": [(7, 23, "PER")]}), ("Kontaktformular: RA Thomas Müller", {"entities": [(16, 32, "PER")]}), ("Impressum - Rechtsanwältin Julia König", {"entities": [(12, 38, "PER")]}), ("Notar und Fachanwalt Dr. Karl Schmidt", {"entities": [(19, 36, "PER")]}), ("Rechtsanwalt Dipl.-Jur. Max Mustermann", {"entities": [(13, 37, "PER")]}), ("RAin Sabine Müller, Fachanwältin für Arbeitsrecht", {"entities": [(5, 18, "PER")]}), ("Impressum: Dr. Peter Becker, LL.M.", {"entities": [(11, 32, "PER")]}), ("RAin Dr. Claudia Meier", {"entities": [(5, 22, "PER")]}), ("Fachanwältin für Familienrecht Julia Brandt", {"entities": [(31, 43, "PER")]}), ("Notarin Dr. Anna Berger", {"entities": [(8, 24, "PER")]}), ("Kontakt: Dr. Laura Fischer", {"entities": [(9, 26, "PER")]}), ("RA Dr. Hans Meier, LL.M.", {"entities": [(3, 19, "PER")]}), ("Impressum: RAin Sabine Becker", {"entities": [(11, 29, "PER")]}), ("Fachanwalt für Strafrecht RA Oliver Weber", {"entities": [(29, 41, "PER")]}), ("Kontaktformular: Dr. Eva Wagner", {"entities": [(18, 32, "PER")]}), ("RA Dr. Karl Fischer", {"entities": [(3, 19, "PER")]}), ("Impressum: Notar Dr. Peter Schmidt", {"entities": [(11, 33, "PER")]}), ("Kontakt: RAin Martina Schneider", {"entities": [(9, 31, "PER")]}), ("Fachanwältin für Arbeitsrecht Anna Meier", {"entities": [(31, 41, "PER")]}), ("Impressum: RA Dr. Hans Meier", {"entities": [(11, 28, "PER")]}), ("Notar Dr. Wolfgang Weber", {"entities": [(6, 23, "PER")]}), ("Kontakt: Rechtsanwältin Dr. Maria Fischer", {"entities": [(9, 36, "PER")]}), ("RA Dr. Michael Schuster", {"entities": [(3, 21, "PER")]}), ("Fachanwalt für Erbrecht RA Dr. Andreas Wagner", {"entities": [(27, 45, "PER")]}), ("Rechtsanwältin Julia König", {"entities": [(15, 27, "PER")]}), ("Rechtsanwalt Dr. Michael Mustermann", {"entities": [(13, 34, "PER")]}), ("Notarin Susanne Bauer", {"entities": [(8, 21, "PER")]}), ("Fachanwalt für Strafrecht Dr. Stefan Weber", {"entities": [(27, 41, "PER")]}), ("Rechtsanwalt Thomas Müller", {"entities": [(13, 27, "PER")]}), ("Kontakt: Rechtsanwalt Dr. Oliver Schulze", {"entities": [(9, 37, "PER")]}), ("Impressum: Rechtsanwältin Claudia Fischer", {"entities": [(11, 39, "PER")]}), ("RA Peter Schmidt", {"entities": [(3, 17, "PER")]}), ("RAin Dr. Martina Meier", {"entities": [(5, 22, "PER")]}), ("Rechtsanwalt Dr. Andreas Müller", {"entities": [(13, 32, "PER")]}), ("Notar Dr. Stefan Fischer", {"entities": [(6, 23, "PER")]}), ("RA Dr. Hans Maier, LL.M.", {"entities": [(3, 18, "PER")]}), ("Fachanwältin für Familienrecht RAin Julia Becker", {"entities": [(35, 47, "PER")]}), ("Kontakt: Dr. Johannes Weber", {"entities": [(9, 26, "PER")]}), ("Rechtsanwalt Dr. Michael Schulze", {"entities": [(13, 34, "PER")]}), ("Impressum: Notar Dr. Hans Maier", {"entities": [(11, 31, "PER")]}), ("RA Dr. Peter Müller", {"entities": [(3, 19, "PER")]}), ("Fachanwalt für Arbeitsrecht Dr. Stefan Weber", {"entities": [(31, 45, "PER")]}), ("Kontaktformular: Dr. Eva Meier", {"entities": [(18, 31, "PER")]}), ("Rechtsanwalt Dr. Hans Mustermann", {"entities": [(14, 32, "PER")]}), ("RA Erika Musterfrau", {"entities": [(3, 18, "PER")]}), ("Notar und Fachanwalt Dr. Karl Schmidt", {"entities": [(19, 36, "PER")]}), ("Rechtsanwalt Dipl.-Jur. Max Mustermann", {"entities": [(13, 37, "PER")]}), ("RAin Sabine Müller, Fachanwältin für Arbeitsrecht", {"entities": [(5, 18, "PER")]}), ("Impressum: Dr. Peter Becker, LL.M.", {"entities": [(11, 32, "PER")]}), ("RAin Dr. Claudia Meier", {"entities": [(5, 22, "PER")]}), ("Fachanwältin für Familienrecht Julia Brandt", {"entities": [(31, 43, "PER")]}), ("Notarin Dr. Anna Berger", {"entities": [(8, 24, "PER")]}), ("Kontakt: Dr. Laura Fischer", {"entities": [(9, 26, "PER")]}), ("RA Dr. Hans Meier, LL.M.", {"entities": [(3, 19, "PER")]}), ("Impressum: RAin Sabine Becker", {"entities": [(11, 29, "PER")]}), ("Fachanwalt für Strafrecht RA Oliver Weber", {"entities": [(29, 41, "PER")]}), ("Kontaktformular: Dr. Eva Wagner", {"entities": [(18, 32, "PER")]}), ("RA Dr. Karl Fischer", {"entities": [(3, 19, "PER")]}), ("Impressum: Notar Dr. Peter Schmidt", {"entities": [(11, 33, "PER")]}), ("Kontakt: RAin Martina Schneider", {"entities": [(9, 31, "PER")]}), ("Fachanwältin für Arbeitsrecht Anna Meier", {"entities": [(31, 41, "PER")]}), ("Impressum: RA Dr. Hans Meier", {"entities": [(11, 28, "PER")]}), ("Notar Dr. Wolfgang Weber", {"entities": [(6, 23, "PER")]}), ("Kontakt: Rechtsanwältin Dr. Maria Fischer", {"entities": [(9, 36, "PER")]}), ("RA Dr. Michael Schuster", {"entities": [(3, 21, "PER")]}), ("Fachanwalt für Erbrecht RA Dr. Andreas Wagner", {"entities": [(27, 45, "PER")]}), ("Rechtsanwältin Julia König", {"entities": [(15, 27, "PER")]}), ("Rechtsanwalt Dr. Michael Mustermann", {"entities": [(13, 34, "PER")]}), ("Notarin Susanne Bauer", {"entities": [(8, 21, "PER")]}), ("Fachanwalt für Strafrecht Dr. Stefan Weber", {"entities": [(27, 41, "PER")]}), ("Rechtsanwalt Thomas Müller", {"entities": [(13, 27, "PER")]}), ("Kontakt: Rechtsanwalt Dr. Oliver Schulze", {"entities": [(9, 37, "PER")]}), ("Impressum: Rechtsanwältin Claudia Fischer", {"entities": [(11, 39, "PER")]}), ("RA Peter Schmidt", {"entities": [(3, 17, "PER")]}), ("RAin Dr. Martina Meier", {"entities": [(5, 22, "PER")]}), ("Rechtsanwalt Dr. Andreas Müller", {"entities": [(13, 32, "PER")]}), ("Notar Dr. Stefan Fischer", {"entities": [(6, 23, "PER")]}), ("RA Dr. Hans Maier, LL.M.", {"entities": [(3, 18, "PER")]}), ("Fachanwältin für Familienrecht RAin Julia Becker", {"entities": [(35, 47, "PER")]}), ("Kontakt: Dr. Johannes Weber", {"entities": [(9, 26, "PER")]}), ("Rechtsanwalt Dr. Michael Schulze", {"entities": [(13, 34, "PER")]}), ("Impressum: Notar Dr. Hans Maier", {"entities": [(11, 31, "PER")]}), ("RA Dr. Peter Müller", {"entities": [(3, 19, "PER")]}), ("Fachanwalt für Arbeitsrecht Dr. Stefan Weber", {"entities": [(31, 45, "PER")]}), ("Kontaktformular: Dr. Eva Meier", {"entities": [(18, 31, "PER")]}),
("kanzlei@anwalt-paderborn.de", {"entities": [(0, 26, "EMAIL")]}), ("info@eikel-partner.de", {"entities": [(0, 20, "EMAIL")]}), ("kontakt@rae-strake.de", {"entities": [(0, 21, "EMAIL")]}), ("ashkan@ra-ashkan.de", {"entities": [(0, 19, "EMAIL")]}), ("kontakt@anwalt-paderborn.com", {"entities": [(0, 28, "EMAIL")]}), ("info@rae-schaefers.de", {"entities": [(0, 19, "EMAIL")]}), ("info@steinertstrafrecht.com", {"entities": [(0, 25, "EMAIL")]}), ("zentrale@kanzlei-am-rosentor.de", {"entities": [(0, 32, "EMAIL")]}), ("kanzlei@anwalt-muster.de", {"entities": [(0, 23, "EMAIL")]}), ("kontakt@kanzlei-xyz.de", {"entities": [(0, 22, "EMAIL")]}), ("info@kanzlei-example.de", {"entities": [(0, 23, "EMAIL")]}), ("support@kanzlei-abc.de", {"entities": [(0, 24, "EMAIL")]}), ("info@kanzlei-123.de", {"entities": [(0, 22, "EMAIL")]}), ("kontakt@anwaltskanzlei.de", {"entities": [(0, 25, "EMAIL")]}), ("kanzlei@recht-und-ordnung.de", {"entities": [(0, 28, "EMAIL")]}), ("anwalt@kanzlei-beispiel.de", {"entities": [(0, 25, "EMAIL")]}), ("info@rechtsanwalt-schmidt.de", {"entities": [(0, 29, "EMAIL")]}), ("contact@lawfirm-example.de", {"entities": [(0, 27, "EMAIL")]}), ("info@lawyer-muster.de", {"entities": [(0, 21, "EMAIL")]}), ("contact@legal-advice.de", {"entities": [(0, 23, "EMAIL")]}), ("kanzlei@rechtsanwaelte-deutschland.de", {"entities": [(0, 35, "EMAIL")]}), ("contact@lawyers-example.de", {"entities": [(0, 26, "EMAIL")]}), ("info@anwaltskanzlei-schmidt.de", {"entities": [(0, 30, "EMAIL")]}), ("support@lawfirm-example.com", {"entities": [(0, 27, "EMAIL")]}), ("info@lawfirm-xyz.com", {"entities": [(0, 19, "EMAIL")]}), ("contact@lawfirm-abc.com", {"entities": [(0, 21, "EMAIL")]}), ("info@legalservices.com", {"entities": [(0, 20, "EMAIL")]}), ("contact@legal-advice.com", {"entities": [(0, 24, "EMAIL")]}), ("info@lawyers-international.com", {"entities": [(0, 28, "EMAIL")]}), ("contact@attorneys-example.com", {"entities": [(0, 28, "EMAIL")]}), ("info@law-firm.com", {"entities": [(0, 15, "EMAIL")]}), ("contact@legal-services.com", {"entities": [(0, 25, "EMAIL")]}), ("info@legal-advice.org", {"entities": [(0, 20, "EMAIL")]}), ("contact@legal-firm.org", {"entities": [(0, 22, "EMAIL")]}), ("info@attorneys-international.org", {"entities": [(0, 31, "EMAIL")]}), ("contact@lawyers-international.org", {"entities": [(0, 32, "EMAIL")]}), ("info@lawyers-firm.org", {"entities": [(0, 21, "EMAIL")]}), ("contact@legal-example.org", {"entities": [(0, 25, "EMAIL")]}), ("info@attorney-services.org", {"entities": [(0, 26, "EMAIL")]}), ("contact@lawfirm-services.org", {"entities": [(0, 27, "EMAIL")]}), ("info@legal-firm-example.org", {"entities": [(0, 27, "EMAIL")]}), ("contact@attorneys-xyz.org", {"entities": [(0, 24, "EMAIL")]}), ("info@legal-advice-xyz.org", {"entities": [(0, 25, "EMAIL")]}), ("contact@lawyer-services.org", {"entities": [(0, 26, "EMAIL")]}), ("info@attorneys-abc.org", {"entities": [(0, 21, "EMAIL")]}), ("contact@lawyer-example.org", {"entities": [(0, 24, "EMAIL")]}), ("info@lawyers-abc.org", {"entities": [(0, 19, "EMAIL")]}), ("contact@legal-services-abc.org", {"entities": [(0, 29, "EMAIL")]}), ("info@lawfirm-example.org", {"entities": [(0, 22, "EMAIL")]}), ("contact@lawfirm-services-xyz.org", {"entities": [(0, 30, "EMAIL")]}), ("info@legal-example-abc.org", {"entities": [(0, 24, "EMAIL")]}), ("contact@lawyers-firm-xyz.org", {"entities": [(0, 27, "EMAIL")]}), ("info@attorneys-example-xyz.org", {"entities": [(0, 29, "EMAIL")]}), ("contact@lawyer-services-xyz.org", {"entities": [(0, 30, "EMAIL")]}), ("info@lawyers-international-xyz.org", {"entities": [(0, 32, "EMAIL")]}), ("contact@legal-advice-example.org", {"entities": [(0, 30, "EMAIL")]}), ("info@attorney-services-xyz.org", {"entities": [(0, 28, "EMAIL")]}), ("contact@lawfirm-xyz.org", {"entities": [(0, 21, "EMAIL")]}), ("info@legal-services-xyz.org", {"entities": [(0, 24, "EMAIL")]}), ("contact@lawyer-example-xyz.org", {"entities": [(0, 29, "EMAIL")]}), ("info@legal-advice-example-xyz.org", {"entities": [(0, 31, "EMAIL")]}), ("contact@lawyers-international-abc.org", {"entities": [(0, 33, "EMAIL")]}), ("info@attorney-example-xyz.org", {"entities": [(0, 27, "EMAIL")]}), ("contact@legal-services-example-xyz.org", {"entities": [(0, 36, "EMAIL")]}), ("info@lawyer-services-abc.org", {"entities": [(0, 27, "EMAIL")]}), ("contact@law-firm-example-xyz.org", {"entities": [(0, 30, "EMAIL")]}), ("info@attorneys-international-xyz.org", {"entities": [(0, 33, "EMAIL")]}), ("contact@lawyers-firm-example-xyz.org", {"entities": [(0, 36, "EMAIL")]}), ("info@lawfirm-abc.org", {"entities": [(0, 16, "EMAIL")]}), ("contact@legal-firm-example-xyz.org", {"entities": [(0, 32, "EMAIL")]}), ("info@attorney-example-abc.org", {"entities": [(0, 27, "EMAIL")]}), ("contact@law-firm-services-abc.org", {"entities": [(0, 32, "EMAIL")]}), ("info@lawyer-example-abc.org", {"entities": [(0, 24, "EMAIL")]}), ("contact@attorneys-services-xyz.org", {"entities": [(0, 32, "EMAIL")]}), ("info@legal-advice-services.org", {"entities": [(0, 28, "EMAIL")]}), ("contact@law-firm-services-xyz.org", {"entities": [(0, 32, "EMAIL")]}), ("info@lawyer-example-xyz.org", {"entities": [(0, 24, "EMAIL")]}), ("contact@attorney-services-example-xyz.org", {"entities": [(0, 38, "EMAIL")]}), ("info@lawyers-firm-example-abc.org", {"entities": [(0, 30, "EMAIL")]}),
("Hans Mustermann", {"entities": [(0, 15, "PER")]}), ("Erika Musterfrau", {"entities": [(0, 16, "PER")]}), ("Max Mustermann", {"entities": [(0, 14, "PER")]}), ("Claudia Meier", {"entities": [(0, 13, "PER")]}), ("Michael Bauer", {"entities": [(0, 13, "PER")]}), ("Laura Fischer", {"entities": [(0, 13, "PER")]}), ("Peter Schmidt", {"entities": [(0, 13, "PER")]}), ("Anna Berger", {"entities": [(0, 11, "PER")]}), ("Johannes Weber", {"entities": [(0, 14, "PER")]}), ("Stefan Weber", {"entities": [(0, 12, "PER")]}), ("Martina Schneider", {"entities": [(0, 16, "PER")]}), ("Andreas Müller", {"entities": [(0, 14, "PER")]}), ("Eva Meier", {"entities": [(0, 9, "PER")]}), ("Hans Maier", {"entities": [(0, 10, "PER")]}), ("Julia Becker", {"entities": [(0, 12, "PER")]}), ("Michael Schulze", {"entities": [(0, 15, "PER")]}), ("Peter Müller", {"entities": [(0, 12, "PER")]}), ("Stefan Fischer", {"entities": [(0, 14, "PER")]}), ("Klaus Weber", {"entities": [(0, 11, "PER")]}), ("Thomas Müller", {"entities": [(0, 13, "PER")]}), ("Maria Fischer", {"entities": [(0, 13, "PER")]}), ("Michael Schuster", {"entities": [(0, 15, "PER")]}), ("Andreas Wagner", {"entities": [(0, 14, "PER")]}), ("Julia König", {"entities": [(0, 11, "PER")]}), ("Michael Mustermann", {"entities": [(0, 18, "PER")]}), ("Susanne Bauer", {"entities": [(0, 13, "PER")]}), ("Oliver Schulze", {"entities": [(0, 13, "PER")]}), ("Claudia Fischer", {"entities": [(0, 15, "PER")]}), ("Peter Schmidt", {"entities": [(0, 13, "PER")]}), ("Martina Meier", {"entities": [(0, 13, "PER")]}), ("Andreas Müller", {"entities": [(0, 14, "PER")]}), ("Stefan Fischer", {"entities": [(0, 14, "PER")]}), ("Hans Maier", {"entities": [(0, 10, "PER")]}), ("Julia Becker", {"entities": [(0, 12, "PER")]}), ("Johannes Weber", {"entities": [(0, 14, "PER")]}), ("Michael Schulze", {"entities": [(0, 15, "PER")]}), ("Peter Müller", {"entities": [(0, 12, "PER")]}), ("Stefan Weber", {"entities": [(0, 12, "PER")]}), ("Klaus Weber", {"entities": [(0, 11, "PER")]}), ("Thomas Müller", {"entities": [(0, 13, "PER")]}), ("Maria Fischer", {"entities": [(0, 13, "PER")]}), ("Michael Schuster", {"entities": [(0, 15, "PER")]}), ("Andreas Wagner", {"entities": [(0, 14, "PER")]}), ("Julia König", {"entities": [(0, 11, "PER")]}), ("Michael Mustermann", {"entities": [(0, 18, "PER")]}), ("Susanne Bauer", {"entities": [(0, 13, "PER")]}), ("Oliver Schulze", {"entities": [(0, 13, "PER")]}), ("Claudia Fischer", {"entities": [(0, 15, "PER")]}), ("Peter Schmidt", {"entities": [(0, 13, "PER")]}), ("Martina Meier", {"entities": [(0, 13, "PER")]}), ("Andreas Müller", {"entities": [(0, 14, "PER")]}), ("Stefan Fischer", {"entities": [(0, 14, "PER")]}), ("Hans Maier", {"entities": [(0, 10, "PER")]}), ("Julia Becker", {"entities": [(0, 12, "PER")]}), ("Johannes Weber", {"entities": [(0, 14, "PER")]}), ("Michael Schulze", {"entities": [(0, 15, "PER")]}), ("Peter Müller", {"entities": [(0, 12, "PER")]}), ("Stefan Weber", {"entities": [(0, 12, "PER")]}), ("Klaus Weber", {"entities": [(0, 11, "PER")]}), ("Thomas Müller", {"entities": [(0, 13, "PER")]}), ("Maria Fischer", {"entities": [(0, 13, "PER")]}), ("Michael Schuster", {"entities": [(0, 15, "PER")]}), ("Andreas Wagner", {"entities": [(0, 14, "PER")]}), ("Julia König", {"entities": [(0, 11, "PER")]}), ("Michael Mustermann", {"entities": [(0, 18, "PER")]}), ("Susanne Bauer", {"entities": [(0, 13, "PER")]}), ("Oliver Schulze", {"entities": [(0, 13, "PER")]}), ("Claudia Fischer", {"entities": [(0, 15, "PER")]}), ("Peter Schmidt", {"entities": [(0, 13, "PER")]}), ("Martina Meier", {"entities": [(0, 13, "PER")]}), ("Andreas Müller", {"entities": [(0, 14, "PER")]}), ("Stefan Fischer", {"entities": [(0, 14, "PER")]}), ("Hans Maier", {"entities": [(0, 10, "PER")]}), ("Julia Becker", {"entities": [(0, 12, "PER")]}), ("Johannes Weber", {"entities": [(0, 14, "PER")]}), ("Michael Schulze", {"entities": [(0, 15, "PER")]}), ("Peter Müller", {"entities": [(0, 12, "PER")]}), ("Stefan Weber", {"entities": [(0, 12, "PER")]}), ("Klaus Weber", {"entities": [(0, 11, "PER")]}), ("Thomas Müller", {"entities": [(0, 13, "PER")]}), ("Maria Fischer", {"entities": [(0, 13, "PER")]}), ("Michael Schuster", {"entities": [(0, 15, "PER")]}), ("Andreas Wagner", {"entities": [(0, 14, "PER")]}), ("Julia König", {"entities": [(0, 11, "PER")]}), ("Michael Mustermann", {"entities": [(0, 18, "PER")]}), ("Susanne Bauer", {"entities": [(0, 13, "PER")]}), ("Oliver Schulze", {"entities": [(0, 13, "PER")]}), ("Claudia Fischer", {"entities": [(0, 15, "PER")]}), ("Peter Schmidt", {"entities": [(0, 13, "PER")]}), ("Martina Meier", {"entities": [(0, 13, "PER")]}), ("Andreas Müller", {"entities": [(0, 14, "PER")]}), ("Stefan Fischer", {"entities": [(0, 14, "PER")]}), ("Hans Maier", {"entities": [(0, 10, "PER")]}), ("Julia Becker", {"entities": [(0, 12, "PER")]}), ("Johannes Weber", {"entities": [(0, 14, "PER")]}), ("Michael Schulze", {"entities": [(0, 15, "PER")]}), ("Peter Müller", {"entities": [(0, 12, "PER")]}), ("Stefan Weber", {"entities": [(0, 12, "PER")]}),
("Martin Schneider", {"entities": [(0, 15, "PER")]}), ("Sophie Wagner", {"entities": [(0, 13, "PER")]}), ("Lukas Becker", {"entities": [(0, 12, "PER")]}), ("Johanna Schmidt", {"entities": [(0, 15, "PER")]}), ("Benjamin Weber", {"entities": [(0, 14, "PER")]}), ("Katrin Bauer", {"entities": [(0, 12, "PER")]}), ("Sebastian Fischer", {"entities": [(0, 17, "PER")]}), ("Eva Schuster", {"entities": [(0, 11, "PER")]}), ("Daniel Wagner", {"entities": [(0, 13, "PER")]}), ("Nina Müller", {"entities": [(0, 10, "PER")]}), ("Felix Weber", {"entities": [(0, 11, "PER")]}), ("Laura Meyer", {"entities": [(0, 11, "PER")]}), ("Tim Becker", {"entities": [(0, 10, "PER")]}), ("Lena Schulz", {"entities": [(0, 10, "PER")]}), ("Jonas Bauer", {"entities": [(0, 11, "PER")]}), ("Paula Fischer", {"entities": [(0, 13, "PER")]}), ("Maximilian Wagner", {"entities": [(0, 17, "PER")]}), ("Clara Meier", {"entities": [(0, 11, "PER")]}), ("Moritz Schmidt", {"entities": [(0, 13, "PER")]}), ("Sarah Weber", {"entities": [(0, 11, "PER")]}), ("Johannes Müller", {"entities": [(0, 15, "PER")]}), ("Marie Schulze", {"entities": [(0, 12, "PER")]}), ("Paul Schuster", {"entities": [(0, 13, "PER")]}), ("Sophie Fischer", {"entities": [(0, 13, "PER")]}), ("Leon Bauer", {"entities": [(0, 10, "PER")]}), ("Lara Weber", {"entities": [(0, 10, "PER")]}), ("Tom Meier", {"entities": [(0, 9, "PER")]}), ("Lina Fischer", {"entities": [(0, 11, "PER")]}), ("David Becker", {"entities": [(0, 12, "PER")]}), ("Lisa Schulze", {"entities": [(0, 11, "PER")]}), ("Jakob Schuster", {"entities": [(0, 14, "PER")]}), ("Emma Weber", {"entities": [(0, 10, "PER")]}), ("Philipp Meier", {"entities": [(0, 13, "PER")]}), ("Hannah Fischer", {"entities": [(0, 13, "PER")]}), ("Julian Bauer", {"entities": [(0, 12, "PER")]}), ("Mia Schulz", {"entities": [(0, 9, "PER")]}), ("Simon Schmidt", {"entities": [(0, 13, "PER")]}), ("Lara Weber", {"entities": [(0, 10, "PER")]}), ("Tim Meyer", {"entities": [(0, 9, "PER")]}), ("Nina Becker", {"entities": [(0, 10, "PER")]}), ("Lukas Schulze", {"entities": [(0, 12, "PER")]}), ("Sophia Schuster", {"entities": [(0, 15, "PER")]}), ("Leon Weber", {"entities": [(0, 10, "PER")]}), ("Mia Meier", {"entities": [(0, 8, "PER")]}), ("Emilia Fischer", {"entities": [(0, 14, "PER")]}), ("Felix Bauer", {"entities": [(0, 11, "PER")]}), ("Anna Schulz", {"entities": [(0, 10, "PER")]}), ("Tom Becker", {"entities": [(0, 10, "PER")]}), ("Lina Schulze", {"entities": [(0, 11, "PER")]}), ("Paul Weber", {"entities": [(0, 10, "PER")]}), ("Katrin Meyer", {"entities": [(0, 12, "PER")]}), ("Benjamin Schmidt", {"entities": [(0, 15, "PER")]}), ("Laura Bauer", {"entities": [(0, 11, "PER")]}), ("Jonas Weber", {"entities": [(0, 11, "PER")]}), ("Eva Fischer", {"entities": [(0, 10, "PER")]}), ("Daniel Schulze", {"entities": [(0, 13, "PER")]}), ("Hannah Schuster", {"entities": [(0, 15, "PER")]}), ("Sebastian Weber", {"entities": [(0, 15, "PER")]}), ("Tim Fischer", {"entities": [(0, 11, "PER")]}), ("Sarah Meier", {"entities": [(0, 11, "PER")]}), ("Martin Bauer", {"entities": [(0, 12, "PER")]}), ("Nina Schulze", {"entities": [(0, 11, "PER")]}), ("Felix Schuster", {"entities": [(0, 13, "PER")]}), ("Sophia Weber", {"entities": [(0, 12, "PER")]}), ("Philipp Meier", {"entities": [(0, 13, "PER")]}), ("Lisa Fischer", {"entities": [(0, 11, "PER")]}), ("Paul Bauer", {"entities": [(0, 10, "PER")]}), ("Sophie Schulz", {"entities": [(0, 12, "PER")]}), ("Maximilian Schmidt", {"entities": [(0, 18, "PER")]}), ("Laura Weber", {"entities": [(0, 11, "PER")]}), ("Johanna Fischer", {"entities": [(0, 15, "PER")]}), ("Sebastian Meyer", {"entities": [(0, 15, "PER")]}), ("Lena Schulze", {"entities": [(0, 11, "PER")]}), ("David Schuster", {"entities": [(0, 13, "PER")]}), ("Klaus Becker", {"entities": [(0, 12, "PER")]}), ("Clara Meier", {"entities": [(0, 11, "PER")]}), ("Jonas Fischer", {"entities": [(0, 13, "PER")]}), ("Eva Bauer", {"entities": [(0, 9, "PER")]}), ("Tim Schulze", {"entities": [(0, 10, "PER")]}), ("Sarah Schuster", {"entities": [(0, 13, "PER")]}), ("Felix Meier", {"entities": [(0, 11, "PER")]}), ("Lukas Fischer", {"entities": [(0, 13, "PER")]}), ("Sophie Weber", {"entities": [(0, 12, "PER")]}), ("Clara Bauer", {"entities": [(0, 11, "PER")]}), ("Tom Schulz", {"entities": [(0, 9, "PER")]}), ("Emma Weber", {"entities": [(0, 10, "PER")]}), ("Julian Fischer", {"entities": [(0, 14, "PER")]}), ("Marie Schulz", {"entities": [(0, 11, "PER")]}), ("David Weber", {"entities": [(0, 11, "PER")]}), ("Lena Becker", {"entities": [(0, 10, "PER")]}),
("Ali Ahmed", {"entities": [(0, 9, "PER")]}), ("Maria Garcia", {"entities": [(0, 12, "PER")]}), ("Chen Wei", {"entities": [(0, 8, "PER")]}), ("Amina Diallo", {"entities": [(0, 12, "PER")]}), ("Carlos Silva", {"entities": [(0, 12, "PER")]}), ("Liam O'Brien", {"entities": [(0, 12, "PER")]}), ("Giulia Rossi", {"entities": [(0, 12, "PER")]}), ("Ahmed Khan", {"entities": [(0, 10, "PER")]}), ("Sara Svensson", {"entities": [(0, 13, "PER")]}), ("Anastasia Ivanov", {"entities": [(0, 16, "PER")]}), ("Juan Martinez", {"entities": [(0, 12, "PER")]}), ("Yuki Nakamura", {"entities": [(0, 13, "PER")]}), ("Aisha Abdi", {"entities": [(0, 10, "PER")]}), ("Marta Kowalski", {"entities": [(0, 14, "PER")]}), ("Mohammed Ali", {"entities": [(0, 12, "PER")]}), ("Hannah Lee", {"entities": [(0, 10, "PER")]}), ("Viktor Petrov", {"entities": [(0, 13, "PER")]}), ("Fatima Hassan", {"entities": [(0, 13, "PER")]}), ("Elias Cohen", {"entities": [(0, 11, "PER")]}), ("Sophia Tanaka", {"entities": [(0, 13, "PER")]}), ("Mehmet Yilmaz", {"entities": [(0, 13, "PER")]}), ("Chloe Nguyen", {"entities": [(0, 11, "PER")]}), ("David Smith", {"entities": [(0, 11, "PER")]}), ("Elena Petrova", {"entities": [(0, 13, "PER")]}), ("Santiago Lopez", {"entities": [(0, 14, "PER")]}), ("Nina Popescu", {"entities": [(0, 12, "PER")]}), ("Arjun Patel", {"entities": [(0, 11, "PER")]}), ("Leila Ben Ali", {"entities": [(0, 13, "PER")]}), ("Igor Petrovic", {"entities": [(0, 13, "PER")]}), ("Isabel Costa", {"entities": [(0, 12, "PER")]}), ("Jin Park", {"entities": [(0, 8, "PER")]}), ("Olga Ivanova", {"entities": [(0, 12, "PER")]}), ("Mateo Garcia", {"entities": [(0, 12, "PER")]}), ("Ming Li", {"entities": [(0, 7, "PER")]}), ("Aisha Ahmed", {"entities": [(0, 11, "PER")]}), ("Viktoria Muller", {"entities": [(0, 15, "PER")]}), ("Fahad Khan", {"entities": [(0, 10, "PER")]}), ("Emily Johnson", {"entities": [(0, 13, "PER")]}), ("Sophia Rodriguez", {"entities": [(0, 15, "PER")]}), ("Khalid Al-Saud", {"entities": [(0, 14, "PER")]}), ("Jana Novak", {"entities": [(0, 10, "PER")]}), ("Ali Rahman", {"entities": [(0, 10, "PER")]}), ("Lucas Martins", {"entities": [(0, 13, "PER")]}), ("Alya Hussein", {"entities": [(0, 12, "PER")]}), ("Julian Schmidt", {"entities": [(0, 14, "PER")]}), ("Daniela Silva", {"entities": [(0, 13, "PER")]}), ("Eva Johansson", {"entities": [(0, 13, "PER")]}), ("Jakub Nowak", {"entities": [(0, 11, "PER")]}), ("Sofia Garcia", {"entities": [(0, 12, "PER")]}), ("Musa Ibrahim", {"entities": [(0, 11, "PER")]}), ("Elif Kaya", {"entities": [(0, 9, "PER")]}), ("Konstantin Ivanov", {"entities": [(0, 17, "PER")]}), ("Mohamed Salah", {"entities": [(0, 13, "PER")]}), ("Aisha Khan", {"entities": [(0, 10, "PER")]}), ("Emily Brown", {"entities": [(0, 11, "PER")]}), ("Hiroshi Sato", {"entities": [(0, 12, "PER")]}), ("Lena Petrova", {"entities": [(0, 12, "PER")]}), ("Marko Nikolic", {"entities": [(0, 13, "PER")]}), ("Olivia Smith", {"entities": [(0, 12, "PER")]}), ("Xia Liu", {"entities": [(0, 7, "PER")]}), ("Isabella Rossi", {"entities": [(0, 14, "PER")]}), ("Nicolas Garcia", {"entities": [(0, 14, "PER")]}), ("Fatima Mohammed", {"entities": [(0, 15, "PER")]}), ("Bartosz Kowalski", {"entities": [(0, 16, "PER")]}), ("Hassan Ali", {"entities": [(0, 10, "PER")]}), ("Tereza Novakova", {"entities": [(0, 15, "PER")]}), ("Ivan Petrov", {"entities": [(0, 10, "PER")]}), ("Ahmed Musa", {"entities": [(0, 10, "PER")]}), ("Anna Nowak", {"entities": [(0, 10, "PER")]}), ("Leonardo Silva", {"entities": [(0, 14, "PER")]}), ("Sophia Kim", {"entities": [(0, 10, "PER")]}), ("David Jones", {"entities": [(0, 11, "PER")]}), ("Rosa Lopez", {"entities": [(0, 10, "PER")]}), ("Ali Hassan", {"entities": [(0, 10, "PER")]}), ("Chen Li", {"entities": [(0, 7, "PER")]}), ("Elena Ivanova", {"entities": [(0, 13, "PER")]}), ("Lucas Oliveira", {"entities": [(0, 14, "PER")]}), ("Amira Abdallah", {"entities": [(0, 14, "PER")]}), ("Svenja Müller", {"entities": [(0, 12, "PER")]}), ("Kenji Takahashi", {"entities": [(0, 15, "PER")]}), ("Marta Fernandez", {"entities": [(0, 15, "PER")]}), ("Viktor Pavlov", {"entities": [(0, 13, "PER")]}), ("Jin Chen", {"entities": [(0, 8, "PER")]}), ("Klara Novak", {"entities": [(0, 11, "PER")]}), ("Hussein Mohamed", {"entities": [(0, 16, "PER")]}), ("David Nguyen", {"entities": [(0, 12, "PER")]}), ("Emilia Petrova", {"entities": [(0, 14, "PER")]}), ("Lars Johansen", {"entities": [(0, 13, "PER")]}), ("Sofia Santos", {"entities": [(0, 12, "PER")]}), ("Nina Müller", {"entities": [(0, 10, "PER")]}), ("Omar Ali", {"entities": [(0, 8, "PER")]}), ("Elif Demir", {"entities": [(0, 10, "PER")]}), ("Isabel Fernandez", {"entities": [(0, 15, "PER")]}),
("Yusuf Ahmed", {"entities": [(0, 11, "PER")]}), ("Hana Kim", {"entities": [(0, 8, "PER")]}), ("Raj Patel", {"entities": [(0, 9, "PER")]}), ("Amina Jallow", {"entities": [(0, 12, "PER")]}), ("Nina Schmidt", {"entities": [(0, 12, "PER")]}), ("Elias Wang", {"entities": [(0, 10, "PER")]}), ("Lucia Rossi", {"entities": [(0, 11, "PER")]}), ("Kofi Mensah", {"entities": [(0, 11, "PER")]}), ("Sara Svensson", {"entities": [(0, 13, "PER")]}), ("Anastasia Ivanov", {"entities": [(0, 16, "PER")]}), ("Juan Martinez", {"entities": [(0, 12, "PER")]}), ("Yuki Nakamura", {"entities": [(0, 13, "PER")]}), ("Aisha Abdi", {"entities": [(0, 10, "PER")]}), ("Marta Kowalski", {"entities": [(0, 14, "PER")]}), ("Mohammed Ali", {"entities": [(0, 12, "PER")]}), ("Hannah Lee", {"entities": [(0, 10, "PER")]}), ("Viktor Petrov", {"entities": [(0, 13, "PER")]}), ("Fatima Hassan", {"entities": [(0, 13, "PER")]}), ("Elias Cohen", {"entities": [(0, 11, "PER")]}), ("Sophia Tanaka", {"entities": [(0, 13, "PER")]}), ("Mehmet Yilmaz", {"entities": [(0, 13, "PER")]}), ("Chloe Nguyen", {"entities": [(0, 11, "PER")]}), ("David Smith", {"entities": [(0, 11, "PER")]}), ("Elena Petrova", {"entities": [(0, 13, "PER")]}), ("Santiago Lopez", {"entities": [(0, 14, "PER")]}), ("Nina Popescu", {"entities": [(0, 12, "PER")]}), ("Arjun Patel", {"entities": [(0, 11, "PER")]}), ("Leila Ben Ali", {"entities": [(0, 13, "PER")]}), ("Igor Petrovic", {"entities": [(0, 13, "PER")]}), ("Isabel Costa", {"entities": [(0, 12, "PER")]}), ("Jin Park", {"entities": [(0, 8, "PER")]}), ("Olga Ivanova", {"entities": [(0, 12, "PER")]}), ("Mateo Garcia", {"entities": [(0, 12, "PER")]}), ("Ming Li", {"entities": [(0, 7, "PER")]}), ("Aisha Ahmed", {"entities": [(0, 11, "PER")]}), ("Viktoria Muller", {"entities": [(0, 15, "PER")]}), ("Fahad Khan", {"entities": [(0, 10, "PER")]}), ("Emily Johnson", {"entities": [(0, 13, "PER")]}), ("Sophia Rodriguez", {"entities": [(0, 15, "PER")]}), ("Khalid Al-Saud", {"entities": [(0, 14, "PER")]}), ("Jana Novak", {"entities": [(0, 10, "PER")]}), ("Ali Rahman", {"entities": [(0, 10, "PER")]}), ("Lucas Martins", {"entities": [(0, 13, "PER")]}), ("Alya Hussein", {"entities": [(0, 12, "PER")]}), ("Julian Schmidt", {"entities": [(0, 14, "PER")]}), ("Daniela Silva", {"entities": [(0, 13, "PER")]}), ("Eva Johansson", {"entities": [(0, 13, "PER")]}), ("Jakub Nowak", {"entities": [(0, 11, "PER")]}), ("Sofia Garcia", {"entities": [(0, 12, "PER")]}), ("Musa Ibrahim", {"entities": [(0, 11, "PER")]}), ("Elif Kaya", {"entities": [(0, 9, "PER")]}), ("Konstantin Ivanov", {"entities": [(0, 17, "PER")]}), ("Mohamed Salah", {"entities": [(0, 13, "PER")]}), ("Aisha Khan", {"entities": [(0, 10, "PER")]}), ("Emily Brown", {"entities": [(0, 11, "PER")]}), ("Hiroshi Sato", {"entities": [(0, 12, "PER")]}), ("Lena Petrova", {"entities": [(0, 12, "PER")]}), ("Marko Nikolic", {"entities": [(0, 13, "PER")]}), ("Olivia Smith", {"entities": [(0, 12, "PER")]}), ("Xia Liu", {"entities": [(0, 7, "PER")]}), ("Isabella Rossi", {"entities": [(0, 14, "PER")]}), ("Nicolas Garcia", {"entities": [(0, 14, "PER")]}), ("Fatima Mohammed", {"entities": [(0, 15, "PER")]}), ("Bartosz Kowalski", {"entities": [(0, 16, "PER")]}), ("Hassan Ali", {"entities": [(0, 10, "PER")]}), ("Tereza Novakova", {"entities": [(0, 15, "PER")]}), ("Ivan Petrov", {"entities": [(0, 10, "PER")]}), ("Ahmed Musa", {"entities": [(0, 10, "PER")]}), ("Anna Nowak", {"entities": [(0, 10, "PER")]}), ("Leonardo Silva", {"entities": [(0, 14, "PER")]}), ("Sophia Kim", {"entities": [(0, 10, "PER")]}), ("David Jones", {"entities": [(0, 11, "PER")]}), ("Rosa Lopez", {"entities": [(0, 10, "PER")]}), ("Ali Hassan", {"entities": [(0, 10, "PER")]}), ("Chen Li", {"entities": [(0, 7, "PER")]}), ("Elena Ivanova", {"entities": [(0, 13, "PER")]}), ("Lucas Oliveira", {"entities": [(0, 14, "PER")]}), ("Amira Abdallah", {"entities": [(0, 14, "PER")]}), ("Svenja Müller", {"entities": [(0, 12, "PER")]}), ("Kenji Takahashi", {"entities": [(0, 15, "PER")]}), ("Marta Fernandez", {"entities": [(0, 15, "PER")]}), ("Viktor Pavlov", {"entities": [(0, 13, "PER")]}), ("Jin Chen", {"entities": [(0, 8, "PER")]}), ("Klara Novak", {"entities": [(0, 11, "PER")]}), ("Hussein Mohamed", {"entities": [(0, 16, "PER")]}), ("David Nguyen", {"entities": [(0, 12, "PER")]}), ("Emilia Petrova", {"entities": [(0, 14, "PER")]}), ("Lars Johansen", {"entities": [(0, 13, "PER")]}), ("Sofia Santos", {"entities": [(0, 12, "PER")]}), ("Nina Müller", {"entities": [(0, 10, "PER")]}), ("Omar Ali", {"entities": [(0, 8, "PER")]}), ("Elif Demir", {"entities": [(0, 10, "PER")]}), ("Isabel Fernandez", {"entities": [(0, 15, "PER")]}), ("Lucas Rossi", {"entities": [(0, 11, "PER")]}), ("Hana Ali", {"entities": [(0, 8, "PER")]}), ("Kim Lee", {"entities": [(0, 7, "PER")]}), ("Rajesh Kumar", {"entities": [(0, 12, "PER")]}), ("Fatoumata Ba", {"entities": [(0, 12, "PER")]}), ("Nadia Schmidt", {"entities": [(0, 13, "PER")]}), ("Elias Zhang", {"entities": [(0, 11, "PER")]}), ("Lucia Neri", {"entities": [(0, 10, "PER")]}), ("Kwame Mensah", {"entities": [(0, 12, "PER")]}), ("Sara Sorensen", {"entities": [(0, 13, "PER")]}), ("Anastasia Ivanova", {"entities": [(0, 17, "PER")]}), ("Juan Morales", {"entities": [(0, 12, "PER")]}), ("Yuki Yamamoto", {"entities": [(0, 13, "PER")]}), ("Amina Abdi", {"entities": [(0, 10, "PER")]}), ("Marta Kowalczyk", {"entities": [(0, 15, "PER")]}), ("Mohammed Hassan", {"entities": [(0, 15, "PER")]}), ("Hannah Tanaka", {"entities": [(0, 13, "PER")]}), ("Viktor Pavlovic", {"entities": [(0, 15, "PER")]}), ("Fatima Ali", {"entities": [(0, 10, "PER")]}), ("Eli Cohen", {"entities": [(0, 9, "PER")]}), ("Sophia Suzuki", {"entities": [(0, 13, "PER")]}), ("Mehmet Yildiz", {"entities": [(0, 13, "PER")]}), ("Chloe Tran", {"entities": [(0, 10, "PER")]}), ("David Johnson", {"entities": [(0, 13, "PER")]}), ("Elena Petrovska", {"entities": [(0, 15, "PER")]}), ("Santiago Ramirez", {"entities": [(0, 16, "PER")]}), ("Nina Popescu", {"entities": [(0, 12, "PER")]}), ("Arjun Reddy", {"entities": [(0, 11, "PER")]}), ("Leila Bouzid", {"entities": [(0, 12, "PER")]}), ("Igor Petrov", {"entities": [(0, 11, "PER")]}), ("Isabel Pereira", {"entities": [(0, 14, "PER")]}), ("Jin Yang", {"entities": [(0, 7, "PER")]}), ("Olga Petrova", {"entities": [(0, 12, "PER")]}), ("Mateo Lopez", {"entities": [(0, 11, "PER")]}), ("Ming Zhao", {"entities": [(0, 9, "PER")]}), ("Aisha Ahmed", {"entities": [(0, 11, "PER")]}), ("Victoria Müller", {"entities": [(0, 15, "PER")]}), ("Fahad Rahman", {"entities": [(0, 12, "PER")]}), ("Emily White", {"entities": [(0, 11, "PER")]}), ("Sophia Hernandez", {"entities": [(0, 16, "PER")]}), ("Khalid Ibrahim", {"entities": [(0, 14, "PER")]}), ("Jana Novakova", {"entities": [(0, 13, "PER")]}), ("Ali Mahmoud", {"entities": [(0, 11, "PER")]}), ("Lucas Ferreira", {"entities": [(0, 14, "PER")]}), ("Alya Mohamed", {"entities": [(0, 12, "PER")]}), ("Julian Muller", {"entities": [(0, 13, "PER")]}), ("Daniela Santos", {"entities": [(0, 14, "PER")]}), ("Eva Andersen", {"entities": [(0, 12, "PER")]}), ("Jakub Novak", {"entities": [(0, 10, "PER")]}), ("Sofia Martins", {"entities": [(0, 13, "PER")]}), ("Musa Yusuf", {"entities": [(0, 10, "PER")]}), ("Elif Demirel", {"entities": [(0, 12, "PER")]}), ("Konstantin Petrov", {"entities": [(0, 17, "PER")]}), ("Mohamed El-Sayed", {"entities": [(0, 15, "PER")]}), ("Aisha Mohamed", {"entities": [(0, 13, "PER")]}), ("Emily Smith", {"entities": [(0, 11, "PER")]}), ("Hiroshi Yamamoto", {"entities": [(0, 15, "PER")]}), ("Lena Petrovska", {"entities": [(0, 13, "PER")]}), ("Marko Jovanovic", {"entities": [(0, 15, "PER")]}), ("Olivia Johnson", {"entities": [(0, 14, "PER")]}), ("Xia Chen", {"entities": [(0, 8, "PER")]}), ("Isabella Fernandes", {"entities": [(0, 17, "PER")]}), ("Nicolas Rodriguez", {"entities": [(0, 17, "PER")]}), ("Fatima Abdallah", {"entities": [(0, 15, "PER")]}), ("Bartosz Nowak", {"entities": [(0, 13, "PER")]}), ("Hassan Mohammed", {"entities": [(0, 15, "PER")]}), ("Tereza Novak", {"entities": [(0, 12, "PER")]}), ("Ivan Ivanov", {"entities": [(0, 10, "PER")]}), ("Ahmed Hassan", {"entities": [(0, 12, "PER")]}), ("Anna Kowalski", {"entities": [(0, 13, "PER")]}), ("Leonardo Santos", {"entities": [(0, 15, "PER")]}), ("Sophia Chen", {"entities": [(0, 10, "PER")]}), ("David Williams", {"entities": [(0, 14, "PER")]}), ("Rosa Garcia", {"entities": [(0, 10, "PER")]}), ("Ali Mohamed", {"entities": [(0, 11, "PER")]}), ("Chen Wu", {"entities": [(0, 7, "PER")]}), ("Elena Petrovska", {"entities": [(0, 15, "PER")]}), ("Lucas Fernandes", {"entities": [(0, 15, "PER")]}), ("Amira El-Sayed", {"entities": [(0, 14, "PER")]}), ("Svenja Schuster", {"entities": [(0, 15, "PER")]}), ("Kenji Yamamoto", {"entities": [(0, 14, "PER")]}), ("Marta Gonzalez", {"entities": [(0, 14, "PER")]}), ("Viktor Ivanov", {"entities": [(0, 13, "PER")]}), ("Jin Liu", {"entities": [(0, 7, "PER")]}), ("Klara Novakova", {"entities": [(0, 14, "PER")]}), ("Hussein El-Sayed", {"entities": [(0, 16, "PER")]}), ("David Tran", {"entities": [(0, 10, "PER")]}), ("Emilia Petrovska", {"entities": [(0, 16, "PER")]}), ("Lars Johansen", {"entities": [(0, 13, "PER")]}), ("Sofia Oliveira", {"entities": [(0, 14, "PER")]}), ("Nina Muller", {"entities": [(0, 10, "PER")]}), ("Omar Hussein", {"entities": [(0, 12, "PER")]}), ("Elif Kaya", {"entities": [(0, 9, "PER")]}), ("Isabel Oliveira", {"entities": [(0, 15, "PER")]}),
("Rechtsanwaltskanzlei Schmidt", {"entities": [(0, 26, "ORG")]}), ("Anwaltsbüro Müller & Partner", {"entities": [(0, 27, "ORG")]}), ("Kanzlei Mustermann & Partner Rechtsanwälte", {"entities": [(0, 41, "ORG")]}), ("Rechtsanwälte Fischer & Söhne GmbH", {"entities": [(0, 32, "ORG")]}), ("Anwaltskanzlei Hansen & Partner mbB", {"entities": [(0, 33, "ORG")]}), ("Rechtsanwaltskanzlei Dr. Becker GmbH", {"entities": [(0, 35, "ORG")]}), ("Kanzlei Müller und Partner mbB", {"entities": [(0, 31, "ORG")]}), ("Kanzlei Berger und Partner", {"entities": [(0, 24, "ORG")]}), ("Rechtsanwälte Dr. Maier & Kollegen", {"entities": [(0, 33, "ORG")]}), ("Rechtsanwaltskanzlei Franke", {"entities": [(0, 26, "ORG")]}), ("Kanzlei Schmidt und Partner mbB", {"entities": [(0, 29, "ORG")]}), ("Anwaltskanzlei Hofmann und Partner", {"entities": [(0, 33, "ORG")]}), ("Kanzlei Schneider & Kollegen", {"entities": [(0, 26, "ORG")]}), ("Rechtsanwaltskanzlei Müller GmbH", {"entities": [(0, 31, "ORG")]}), ("Anwaltskanzlei Schulze & Kollegen GbR", {"entities": [(0, 35, "ORG")]}), ("Kanzlei Weber & Partner", {"entities": [(0, 22, "ORG")]}), ("Rechtsanwälte Fischer und Partner", {"entities": [(0, 32, "ORG")]}), ("Kanzlei Braun und Kollegen", {"entities": [(0, 26, "ORG")]}), ("Rechtsanwälte Meier & Söhne AG", {"entities": [(0, 32, "ORG")]}), ("Anwaltskanzlei Schröder mbH", {"entities": [(0, 27, "ORG")]}), ("Kanzlei Krüger & Partner", {"entities": [(0, 22, "ORG")]}), ("Rechtsanwälte Wolf und Kollegen", {"entities": [(0, 32, "ORG")]}), ("Kanzlei Vogel & Partner", {"entities": [(0, 21, "ORG")]}), ("Rechtsanwälte Lutz und Partner", {"entities": [(0, 31, "ORG")]}), ("Kanzlei Beck & Kollegen", {"entities": [(0, 21, "ORG")]}), ("Kanzlei Richter und Partner mbB", {"entities": [(0, 30, "ORG")]}), ("Rechtsanwälte Wagner & Partner", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Jung und Kollegen", {"entities": [(0, 24, "ORG")]}), ("Rechtsanwälte Hoffmann und Partner", {"entities": [(0, 35, "ORG")]}), ("Rechtsanwaltskanzlei Kaiser GmbH", {"entities": [(0, 32, "ORG")]}), ("Anwaltskanzlei Scholz & Partner mbB", {"entities": [(0, 33, "ORG")]}), ("Kanzlei Schmitt und Partner", {"entities": [(0, 27, "ORG")]}), ("Rechtsanwälte Schmid & Kollegen", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Lehmann & Partner", {"entities": [(0, 25, "ORG")]}), ("Rechtsanwälte Schulze und Partner", {"entities": [(0, 33, "ORG")]}), ("Kanzlei Fuchs und Partner mbB", {"entities": [(0, 28, "ORG")]}), ("Anwaltskanzlei Herrmann & Kollegen", {"entities": [(0, 34, "ORG")]}), ("Rechtsanwaltskanzlei Neumann", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Meyer und Partner", {"entities": [(0, 24, "ORG")]}), ("Rechtsanwälte Schmidt & Partner", {"entities": [(0, 30, "ORG")]}), ("Anwaltskanzlei Bauer & Kollegen", {"entities": [(0, 31, "ORG")]}), ("Kanzlei Krüger und Partner", {"entities": [(0, 25, "ORG")]}), ("Rechtsanwälte Maier & Kollegen", {"entities": [(0, 31, "ORG")]}), ("Anwaltskanzlei Braun & Partner", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Richter und Partner mbB", {"entities": [(0, 30, "ORG")]}), ("Rechtsanwälte Wagner & Partner", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Jung und Kollegen", {"entities": [(0, 24, "ORG")]}), ("Rechtsanwälte Hoffmann und Partner", {"entities": [(0, 35, "ORG")]}), ("Rechtsanwaltskanzlei Neumann", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Meyer und Partner", {"entities": [(0, 24, "ORG")]}), ("Rechtsanwälte Schmidt & Partner", {"entities": [(0, 30, "ORG")]}), ("Anwaltskanzlei Bauer & Kollegen", {"entities": [(0, 31, "ORG")]}), ("Kanzlei Krüger und Partner", {"entities": [(0, 25, "ORG")]}), ("Rechtsanwälte Maier & Kollegen", {"entities": [(0, 31, "ORG")]}), ("Anwaltskanzlei Braun & Partner", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Richter und Partner mbB", {"entities": [(0, 30, "ORG")]}), ("Rechtsanwälte Wagner & Partner", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Jung und Kollegen", {"entities": [(0, 24, "ORG")]}), ("Rechtsanwälte Hoffmann und Partner", {"entities": [(0, 35, "ORG")]}), ("Rechtsanwaltskanzlei Neumann", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Meyer und Partner", {"entities": [(0, 24, "ORG")]}), ("Rechtsanwälte Schmidt & Partner", {"entities": [(0, 30, "ORG")]}), ("Anwaltskanzlei Bauer & Kollegen", {"entities": [(0, 31, "ORG")]}), ("Kanzlei Krüger und Partner", {"entities": [(0, 25, "ORG")]}), ("Rechtsanwälte Maier & Kollegen", {"entities": [(0, 31, "ORG")]}), ("Anwaltskanzlei Braun & Partner", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Richter und Partner mbB", {"entities": [(0, 30, "ORG")]}), ("Rechtsanwälte Wagner & Partner", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Jung und Kollegen", {"entities": [(0, 24, "ORG")]}), ("Rechtsanwälte Hoffmann und Partner", {"entities": [(0, 35, "ORG")]}), ("Rechtsanwaltskanzlei Neumann", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Meyer und Partner", {"entities": [(0, 24, "ORG")]}), ("Rechtsanwälte Schmidt & Partner", {"entities": [(0, 30, "ORG")]}), ("Anwaltskanzlei Bauer & Kollegen", {"entities": [(0, 31, "ORG")]}), ("Kanzlei Krüger und Partner", {"entities": [(0, 25, "ORG")]}), ("Rechtsanwälte Maier & Kollegen", {"entities": [(0, 31, "ORG")]}), ("Anwaltskanzlei Braun & Partner", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Richter und Partner mbB", {"entities": [(0, 30, "ORG")]}), ("Rechtsanwälte Wagner & Partner", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Jung und Kollegen", {"entities": [(0, 24, "ORG")]}), ("Rechtsanwälte Hoffmann und Partner", {"entities": [(0, 35, "ORG")]}), ("Rechtsanwaltskanzlei Neumann", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Meyer und Partner", {"entities": [(0, 24, "ORG")]}), ("Rechtsanwälte Schmidt & Partner", {"entities": [(0, 30, "ORG")]}), ("Anwaltskanzlei Bauer & Kollegen", {"entities": [(0, 31, "ORG")]}), ("Kanzlei Krüger und Partner", {"entities": [(0, 25, "ORG")]}), ("Rechtsanwälte Maier & Kollegen", {"entities": [(0, 31, "ORG")]}), ("Anwaltskanzlei Braun & Partner", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Richter und Partner mbB", {"entities": [(0, 30, "ORG")]}), ("Rechtsanwälte Wagner & Partner", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Jung und Kollegen", {"entities": [(0, 24, "ORG")]}), ("Rechtsanwälte Hoffmann und Partner", {"entities": [(0, 35, "ORG")]}), ("Rechtsanwaltskanzlei Neumann", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Meyer und Partner", {"entities": [(0, 24, "ORG")]}), ("Rechtsanwälte Schmidt & Partner", {"entities": [(0, 30, "ORG")]}), ("Anwaltskanzlei Bauer & Kollegen", {"entities": [(0, 31, "ORG")]}), ("Kanzlei Krüger und Partner", {"entities": [(0, 25, "ORG")]}), ("Rechtsanwälte Maier & Kollegen", {"entities": [(0, 31, "ORG")]}), ("Anwaltskanzlei Braun & Partner", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Richter und Partner mbB", {"entities": [(0, 30, "ORG")]}), ("Rechtsanwälte Wagner & Partner", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Jung und Kollegen", {"entities": [(0, 24, "ORG")]}), ("Rechtsanwälte Hoffmann und Partner", {"entities": [(0, 35, "ORG")]}), ("Rechtsanwaltskanzlei Neumann", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Meyer und Partner", {"entities": [(0, 24, "ORG")]}), ("Rechtsanwälte Schmidt & Partner", {"entities": [(0, 30, "ORG")]}), ("Anwaltskanzlei Bauer & Kollegen", {"entities": [(0, 31, "ORG")]}), ("Kanzlei Krüger und Partner", {"entities": [(0, 25, "ORG")]}), ("Rechtsanwälte Maier & Kollegen", {"entities": [(0, 31, "ORG")]}), ("Anwaltskanzlei Braun & Partner", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Richter und Partner mbB", {"entities": [(0, 30, "ORG")]}), ("Rechtsanwälte Wagner & Partner", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Jung und Kollegen", {"entities": [(0, 24, "ORG")]}), ("Rechtsanwälte Hoffmann und Partner", {"entities": [(0, 35, "ORG")]}), ("Rechtsanwaltskanzlei Neumann", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Meyer und Partner", {"entities": [(0, 24, "ORG")]}), ("Rechtsanwälte Schmidt & Partner", {"entities": [(0, 30, "ORG")]}), ("Anwaltskanzlei Bauer & Kollegen", {"entities": [(0, 31, "ORG")]}), ("Kanzlei Krüger und Partner", {"entities": [(0, 25, "ORG")]}), ("Rechtsanwälte Maier & Kollegen", {"entities": [(0, 31, "ORG")]}), ("Anwaltskanzlei Braun & Partner", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Richter und Partner mbB", {"entities": [(0, 30, "ORG")]}), ("Rechtsanwälte Wagner & Partner", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Jung und Kollegen", {"entities": [(0, 24, "ORG")]}), ("Rechtsanwälte Hoffmann und Partner", {"entities": [(0, 35, "ORG")]}), ("Rechtsanwaltskanzlei Neumann", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Meyer und Partner", {"entities": [(0, 24, "ORG")]}), ("Rechtsanwälte Schmidt & Partner", {"entities": [(0, 30, "ORG")]}), ("Anwaltskanzlei Bauer & Kollegen", {"entities": [(0, 31, "ORG")]}), ("Kanzlei Krüger und Partner", {"entities": [(0, 25, "ORG")]}), ("Rechtsanwälte Maier & Kollegen", {"entities": [(0, 31, "ORG")]}), ("Anwaltskanzlei Braun & Partner", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Richter und Partner mbB", {"entities": [(0, 30, "ORG")]}), ("Rechtsanwälte Wagner & Partner", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Jung und Kollegen", {"entities": [(0, 24, "ORG")]}), ("Rechtsanwälte Hoffmann und Partner", {"entities": [(0, 35, "ORG")]}),
("Law Offices of Smith & Associates", {"entities": [(0, 32, "ORG")]}), ("Johnson & Partners Law Firm", {"entities": [(0, 26, "ORG")]}), ("Advocates R. Garcia & Co.", {"entities": [(0, 25, "ORG")]}), ("Miller Legal Services", {"entities": [(0, 22, "ORG")]}), ("Brown, Green & Lee Attorneys", {"entities": [(0, 31, "ORG")]}), ("Nguyen & Tran Solicitors", {"entities": [(0, 24, "ORG")]}), ("Kanzlei Müller & Partner", {"entities": [(0, 24, "ORG")]}), ("Law Group of Chan & Associates", {"entities": [(0, 30, "ORG")]}), ("Advokatfirmaet Hansen", {"entities": [(0, 21, "ORG")]}), ("Rechtsanwälte Bauer & Schmidt", {"entities": [(0, 30, "ORG")]}), ("Kanzlei Mustermann & Kollegen", {"entities": [(0, 28, "ORG")]}), ("Firm of White, Black & Grey", {"entities": [(0, 28, "ORG")]}), ("Advokatfirman Johansson", {"entities": [(0, 24, "ORG")]}), ("Law Office of Smith and Wesson", {"entities": [(0, 30, "ORG")]}), ("Rechtsanwalt Dr. Schmidt & Co.", {"entities": [(0, 31, "ORG")]}), ("Legal Advisors of Martinez & Co.", {"entities": [(0, 32, "ORG")]}), ("Firma Rechtsanwälte Maier & Müller", {"entities": [(0, 36, "ORG")]}), ("Law Corporation of Patel & Patel", {"entities": [(0, 31, "ORG")]}), ("Anwaltskanzlei Schulz & Partner", {"entities": [(0, 30, "ORG")]}), ("Juristenbüro Fischer", {"entities": [(0, 20, "ORG")]}), ("Legal Team of Lopez & Hernandez", {"entities": [(0, 31, "ORG")]}), ("Solicitors Green & Brown", {"entities": [(0, 24, "ORG")]}), ("Law Chambers of Dr. Ibrahim", {"entities": [(0, 28, "ORG")]}), ("Advokatkontor Andersen", {"entities": [(0, 24, "ORG")]}), ("Legal Associates of Wang & Kim", {"entities": [(0, 30, "ORG")]}), ("Anwaltskanzlei Hoffmann & Kollegen", {"entities": [(0, 33, "ORG")]}), ("Law Offices of Johnson & Johnson", {"entities": [(0, 31, "ORG")]}), ("Rechtsanwälte Krüger & Meyer", {"entities": [(0, 30, "ORG")]}), ("Legal Firm of Schmidt & Brown", {"entities": [(0, 30, "ORG")]}), ("Advokatbyrå Larsen", {"entities": [(0, 18, "ORG")]}), ("Firma Rechtsanwälte Fischer & Weber", {"entities": [(0, 36, "ORG")]}), ("Law Office of Mustafa & Ahmed", {"entities": [(0, 29, "ORG")]}), ("Rechtsanwälte Becker & Partner", {"entities": [(0, 30, "ORG")]}), ("Anwaltskanzlei Schuster & Partner", {"entities": [(0, 32, "ORG")]}), ("Legal Experts of Müller & Söhne", {"entities": [(0, 30, "ORG")]}), ("Firma Rechtsanwalt Schneider", {"entities": [(0, 29, "ORG")]}), ("Advokatfirmaet Pedersen", {"entities": [(0, 24, "ORG")]}), ("Law Practice of Garcia & Sons", {"entities": [(0, 28, "ORG")]}), ("Juristenbüro Schmidt & Meier", {"entities": [(0, 29, "ORG")]}), ("Legal Partners of Nguyen & Lee", {"entities": [(0, 30, "ORG")]}), ("Anwaltskanzlei Jung & Partner", {"entities": [(0, 28, "ORG")]}), ("Law Firm of Dr. Hofmann & Co.", {"entities": [(0, 29, "ORG")]}), ("Rechtsanwälte Braun & Kollegen", {"entities": [(0, 31, "ORG")]}), ("Advokatur von Kraus & Müller", {"entities": [(0, 28, "ORG")]}), ("Firma Juristenbüro Hoffmann", {"entities": [(0, 29, "ORG")]}), ("Legal Office of White & Green", {"entities": [(0, 29, "ORG")]}), ("Advokatfirmaet Nilsen", {"entities": [(0, 23, "ORG")]}), ("Law Services of Martinez & Co.", {"entities": [(0, 30, "ORG")]}), ("Anwaltskanzlei Fuchs & Partner", {"entities": [(0, 29, "ORG")]}), ("Legal Associates of Kim & Park", {"entities": [(0, 29, "ORG")]}), ("Advokater Dr. Maier & Kollegen", {"entities": [(0, 31, "ORG")]}), ("Rechtsanwälte Schröder & Partner", {"entities": [(0, 33, "ORG")]}), ("Law Office of Brown & White", {"entities": [(0, 27, "ORG")]}), ("Juristenbüro Weber & Meyer", {"entities": [(0, 28, "ORG")]}), ("Firma Rechtsanwälte Schneider & Müller", {"entities": [(0, 39, "ORG")]}), ("Advokatfirmaet Svensson", {"entities": [(0, 24, "ORG")]}), ("Law Practice of Dr. Smith & Co.", {"entities": [(0, 31, "ORG")]}), ("Anwaltskanzlei Schwarz & Partner", {"entities": [(0, 32, "ORG")]}), ("Legal Experts of Brown & Green", {"entities": [(0, 30, "ORG")]}), ("Firma Rechtsanwalt Schmidt & Partner", {"entities": [(0, 36, "ORG")]}), ("Advokatbyrå Eriksson", {"entities": [(0, 19, "ORG")]}), ("Law Office of Green & Johnson", {"entities": [(0, 30, "ORG")]}), ("Rechtsanwälte Schuster & Söhne", {"entities": [(0, 31, "ORG")]}), ("Anwaltskanzlei Schwarz & Kollegen", {"entities": [(0, 34, "ORG")]}), ("Legal Associates of Dr. Chen", {"entities": [(0, 30, "ORG")]}), ("Law Chambers of Martinez & Co.", {"entities": [(0, 31, "ORG")]}), ("Firma Juristenbüro Maier & Müller", {"entities": [(0, 35, "ORG")]}), ("Advokatfirmaet Jensen", {"entities": [(0, 22, "ORG")]}), ("Law Practice of Nguyen & Lee", {"entities": [(0, 28, "ORG")]}), ("Rechtsanwälte Weber & Partner", {"entities": [(0, 29, "ORG")]}), ("Anwaltskanzlei Braun & Kollegen", {"entities": [(0, 32, "ORG")]}), ("Legal Experts of Smith & Brown", {"entities": [(0, 30, "ORG")]}), ("Firma Rechtsanwälte Schulz & Partner", {"entities": [(0, 37, "ORG")]}), ("Advokatkontor Johansson", {"entities": [(0, 24, "ORG")]}), ("Law Offices of Dr. Johnson & Co.", {"entities": [(0, 33, "ORG")]}), ("Rechtsanwälte Meier & Schmidt", {"entities": [(0, 30, "ORG")]}), ("Anwaltskanzlei Schulz & Kollegen", {"entities": [(0, 33, "ORG")]}), ("Legal Firm of Garcia & Martinez", {"entities": [(0, 31, "ORG")]}), ("Firma Juristenbüro Fischer & Weber", {"entities": [(0, 36, "ORG")]}), ("Advokatfirmaet Larsen", {"entities": [(0, 22, "ORG")]}), ("Law Office of Dr. Ahmed & Co.", {"entities": [(0, 30, "ORG")]}), ("Rechtsanwälte Bauer & Hoffmann", {"entities": [(0, 31, "ORG")]}), ("Anwaltskanzlei Schuster & Söhne", {"entities": [(0, 32, "ORG")]}), ("Legal Partners of Wang & Lee", {"entities": [(0, 29, "ORG")]}), ("Firma Rechtsanwalt Müller & Partner", {"entities": [(0, 36, "ORG")]}), ("Advokatbyrå Karlsson", {"entities": [(0, 20, "ORG")]}), ("Law Practice of Smith & Johnson", {"entities": [(0, 31, "ORG")]}), ("Rechtsanwälte Fischer & Meyer", {"entities": [(0, 30, "ORG")]}), ("Anwaltskanzlei Braun & Söhne", {"entities": [(0, 29, "ORG")]}), ("Legal Firm of Dr. Schmidt & Co.", {"entities": [(0, 31, "ORG")]}), ("Firma Rechtsanwälte Schulze & Partner", {"entities": [(0, 38, "ORG")]}), ("Advokatkontor Nilsen", {"entities": [(0, 21, "ORG")]}), ("Law Practice of Garcia & Hernandez", {"entities": [(0, 33, "ORG")]}), ("Rechtsanwälte Weber & Hoffmann", {"entities": [(0, 31, "ORG")]}), ("Anwaltskanzlei Schulz & Kollegen", {"entities": [(0, 33, "ORG")]}), ("Legal Firm of Kim & Park", {"entities": [(0, 25, "ORG")]}), ("Firma Rechtsanwalt Schneider & Söhne", {"entities": [(0, 38, "ORG")]}), ("Advokatfirmaet Pedersen", {"entities": [(0, 24, "ORG")]}), ("Law Office of Dr. Smith & Co.", {"entities": [(0, 29, "ORG")]}), ("Rechtsanwälte Bauer & Kollegen", {"entities": [(0, 30, "ORG")]}), ("Anwaltskanzlei Braun & Partner", {"entities": [(0, 29, "ORG")]}), ("Legal Firm of Dr. Maier & Co.", {"entities": [(0, 30, "ORG")]}), ("Firma Rechtsanwälte Schuster & Partner", {"entities": [(0, 38, "ORG")]}), ("Advokatbyrå Eriksson", {"entities": [(0, 19, "ORG")]}), ("Law Office of Johnson & Johnson", {"entities": [(0, 31, "ORG")]}), ("Rechtsanwälte Fischer & Partner", {"entities": [(0, 31, "ORG")]}), ("Anwaltskanzlei Schuster & Meyer", {"entities": [(0, 30, "ORG")]}), ("Legal Experts of Brown & Lee", {"entities": [(0, 27, "ORG")]}), ("Firma Rechtsanwalt Schmidt & Müller", {"entities": [(0, 37, "ORG")]}), ("Advokatkontor Andersen", {"entities": [(0, 23, "ORG")]}), ("Law Practice of Kim & Johnson", {"entities": [(0, 30, "ORG")]}), ("Rechtsanwälte Weber & Söhne", {"entities": [(0, 29, "ORG")]}), ("Anwaltskanzlei Schuster & Partner", {"entities": [(0, 32, "ORG")]}), ("Legal Experts of Dr. Maier", {"entities": [(0, 28, "ORG")]}), ("Firma Juristenbüro Schulze & Partner", {"entities": [(0, 37, "ORG")]}), ("Advokatfirmaet Jensen", {"entities": [(0, 22, "ORG")]}), ("Law Office of Smith & Brown", {"entities": [(0, 26, "ORG")]}), ("Rechtsanwälte Meier & Söhne", {"entities": [(0, 29, "ORG")]}), ("Anwaltskanzlei Schulz & Söhne", {"entities": [(0, 29, "ORG")]}), ("Legal Firm of Dr. Johnson & Co.", {"entities": [(0, 33, "ORG")]}), ("Firma Rechtsanwälte Maier & Müller", {"entities": [(0, 36, "ORG")]}), ("Advokatbyrå Svensson", {"entities": [(0, 20, "ORG")]}), ("Law Office of Garcia & Smith", {"entities": [(0, 28, "ORG")]}), ("Rechtsanwälte Fischer & Weber", {"entities": [(0, 29, "ORG")]}), ("Anwaltskanzlei Braun & Söhne", {"entities": [(0, 29, "ORG")]}), ("Legal Firm of Kim & Nguyen", {"entities": [(0, 27, "ORG")]}), ("Firma Rechtsanwalt Müller & Partner", {"entities": [(0, 36, "ORG")]}), ("Advokatkontor Larsen", {"entities": [(0, 19, "ORG")]}), ("Law Practice of Johnson & Brown", {"entities": [(0, 31, "ORG")]}), ("Rechtsanwälte Schuster & Meyer", {"entities": [(0, 30, "ORG")]}), ("Anwaltskanzlei Schwarz & Söhne", {"entities": [(0, 29, "ORG")]}), ("Legal Experts of Brown & Johnson", {"entities": [(0, 32, "ORG")]}), ("Firma Juristenbüro Schulz & Weber", {"entities": [(0, 33, "ORG")]}), ("Advokatfirmaet Nilsen", {"entities": [(0, 21, "ORG")]}), ("Law Office of Smith & Johnson", {"entities": [(0, 29, "ORG")]}), ("Rechtsanwälte Braun & Söhne", {"entities": [(0, 29, "ORG")]}), ("Anwaltskanzlei Schulz & Meyer", {"entities": [(0, 30, "ORG")]}), ("Legal Firm of Dr. Johnson & Co.", {"entities": [(0, 33, "ORG")]}), ("Firma Rechtsanwälte Schulze & Müller", {"entities": [(0, 38, "ORG")]}), ("Advokatbyrå Johansson", {"entities": [(0, 21, "ORG")]}), ("Law Office of Garcia & Johnson", {"entities": [(0, 30, "ORG")]}), ("Rechtsanwälte Fischer & Schmidt", {"entities": [(0, 31, "ORG")]}), ("Anwaltskanzlei Braun & Partner", {"entities": [(0, 29, "ORG")]}), ("Legal Firm of Dr. Kim & Co.", {"entities": [(0, 26, "ORG")]}), ("Firma Juristenbüro Meier & Müller", {"entities": [(0, 35, "ORG")]}), ("Advokatkontor Andersen", {"entities": [(0, 23, "ORG")]}), ("Law Practice of Johnson & Smith", {"entities": [(0, 31, "ORG")]}), ("Rechtsanwälte Schuster & Partner", {"entities": [(0, 33, "ORG")]}), ("Anwaltskanzlei Schulz & Partner", {"entities": [(0, 30, "ORG")]}), ("Legal Experts of Dr. Maier & Co.", {"entities": [(0, 32, "ORG")]}), ("Firma Rechtsanwälte Müller & Weber", {"entities": [(0, 36, "ORG")]}), ("Advokatfirmaet Pedersen", {"entities": [(0, 24, "ORG")]}), ("Law Office of Johnson & Lee", {"entities": [(0, 27, "ORG")]}), ("Rechtsanwälte Bauer & Söhne", {"entities": [(0, 28, "ORG")]}), ("Anwaltskanzlei Schulz & Kollegen", {"entities": [(0, 33, "ORG")]}), ("Legal Firm of Dr. Johnson & Brown", {"entities": [(0, 34, "ORG")]}), ("Firma Rechtsanwälte Meier & Schmidt", {"entities": [(0, 36, "ORG")]}), ("Advokatbyrå Eriksson", {"entities": [(0, 19, "ORG")]}), ("Law Office of Smith & Martinez", {"entities": [(0, 29, "ORG")]}), ("Rechtsanwälte Schuster & Kollegen", {"entities": [(0, 33, "ORG")]}), ("Anwaltskanzlei Schulz & Weber", {"entities": [(0, 29, "ORG")]}), ("Legal Firm of Dr. Chen & Co.", {"entities": [(0, 27, "ORG")]}), ("Firma Juristenbüro Meier & Weber", {"entities": [(0, 34, "ORG")]}), ("Advokatfirmaet Jensen", {"entities": [(0, 22, "ORG")]}), ("Law Practice of Smith & Nguyen", {"entities": [(0, 30, "ORG")]}), ("Rechtsanwälte Braun & Partner", {"entities": [(0, 30, "ORG")]}), ("Anwaltskanzlei Schulz & Söhne", {"entities": [(0, 29, "ORG")]}), ("Legal Firm of Dr. Johnson & Martinez", {"entities": [(0, 37, "ORG")]}), ("Firma Rechtsanwälte Schulze & Weber", {"entities": [(0, 37, "ORG")]}), ("Advokatkontor Nilsen", {"entities": [(0, 21, "ORG")]}), ("Law Office of Garcia & Nguyen", {"entities": [(0, 29, "ORG")]}), ("Rechtsanwälte Braun & Schmidt", {"entities": [(0, 30, "ORG")]}), ("Anwaltskanzlei Schulz & Meyer", {"entities": [(0, 30, "ORG")]}), ("Legal Firm of Dr. Maier & Partner", {"entities": [(0, 33, "ORG")]}), ("Firma Juristenbüro Fischer & Weber", {"entities": [(0, 36, "ORG")]}), ("Advokatfirmaet Larsen", {"entities": [(0, 22, "ORG")]}), ("Law Practice of Garcia & Johnson", {"entities": [(0, 32, "ORG")]}), ("Rechtsanwälte Fischer & Söhne", {"entities": [(0, 30, "ORG")]}), ("Anwaltskanzlei Schulz & Söhne", {"entities": [(0, 29, "ORG")]}), ("Legal Firm of Dr. Kim & Johnson", {"entities": [(0, 32, "ORG")]}), ("Firma Rechtsanwalt Meier & Müller", {"entities": [(0, 35, "ORG")]}), ("Advokatkontor Eriksson", {"entities": [(0, 21, "ORG")]}), ("Law Office of Johnson & Smith", {"entities": [(0, 29, "ORG")]}), ("Rechtsanwälte Braun & Weber", {"entities": [(0, 29, "ORG")]}), ("Anwaltskanzlei Schulz & Weber", {"entities": [(0, 29, "ORG")]}), ("Legal Firm of Dr. Schmidt & Johnson", {"entities": [(0, 34, "ORG")]}), ("Firma Rechtsanwalt Fischer & Partner", {"entities": [(0, 36, "ORG")]}),
("Liam O'Brien", {"entities": [(0, 12, "PER")]}),
    ("Marta Kowalski", {"entities": [(0, 14, "PER")]}),
    ("Chen Wei", {"entities": [(0, 8, "PER")]}),
    ("Amina Diallo", {"entities": [(0, 12, "PER")]}),
    ("Carlos Silva", {"entities": [(0, 12, "PER")]}),
    ("Giulia Rossi", {"entities": [(0, 12, "PER")]}),
    ("Ahmed Khan", {"entities": [(0, 10, "PER")]}),
    ("Sara Svensson", {"entities": [(0, 13, "PER")]}),
    ("Anastasia Ivanov", {"entities": [(0, 16, "PER")]}),
    ("Juan Martinez", {"entities": [(0, 12, "PER")]}),
    ("Yuki Nakamura", {"entities": [(0, 13, "PER")]}),
    ("Aisha Abdi", {"entities": [(0, 10, "PER")]}),
    ("Mohammed Ali", {"entities": [(0, 12, "PER")]}),
    ("Hannah Lee", {"entities": [(0, 10, "PER")]}),
    ("Viktor Petrov", {"entities": [(0, 13, "PER")]}),
    ("Fatima Hassan", {"entities": [(0, 13, "PER")]}),
    ("Elias Cohen", {"entities": [(0, 11, "PER")]}),
    ("Sophia Tanaka", {"entities": [(0, 13, "PER")]}),
    ("Mehmet Yilmaz", {"entities": [(0, 13, "PER")]}),
    ("Chloe Nguyen", {"entities": [(0, 11, "PER")]}),
    ("David Smith", {"entities": [(0, 11, "PER")]}),
    ("Elena Petrova", {"entities": [(0, 13, "PER")]}),
    ("Santiago Lopez", {"entities": [(0, 14, "PER")]}),
    ("Nina Popescu", {"entities": [(0, 12, "PER")]}),
    ("Arjun Patel", {"entities": [(0, 11, "PER")]}),
    ("Leila Ben Ali", {"entities": [(0, 13, "PER")]}),
    ("Igor Petrovic", {"entities": [(0, 13, "PER")]}),
    ("Isabel Costa", {"entities": [(0, 12, "PER")]}),
    ("Jin Park", {"entities": [(0, 8, "PER")]}),
    ("Olga Ivanova", {"entities": [(0, 12, "PER")]}),
    ("Mateo Garcia", {"entities": [(0, 12, "PER")]}),
    ("Ming Li", {"entities": [(0, 7, "PER")]}),
    ("Aisha Ahmed", {"entities": [(0, 11, "PER")]}),
    ("Viktoria Muller", {"entities": [(0, 15, "PER")]}),
    ("Fahad Khan", {"entities": [(0, 10, "PER")]}),
    ("Emily Johnson", {"entities": [(0, 13, "PER")]}),
    ("Sophia Rodriguez", {"entities": [(0, 15, "PER")]}),
    ("Khalid Al-Saud", {"entities": [(0, 14, "PER")]}),
    ("Jana Novak", {"entities": [(0, 10, "PER")]}),
    ("Ali Rahman", {"entities": [(0, 10, "PER")]}),
    ("Lucas Martins", {"entities": [(0, 13, "PER")]}),
    ("Alya Hussein", {"entities": [(0, 12, "PER")]}),
    ("Julian Schmidt", {"entities": [(0, 14, "PER")]}),
    ("Daniela Silva", {"entities": [(0, 13, "PER")]}),
    ("Eva Johansson", {"entities": [(0, 13, "PER")]}),
    ("Jakub Nowak", {"entities": [(0, 11, "PER")]}),
    ("Sofia Garcia", {"entities": [(0, 12, "PER")]}),
    ("Musa Ibrahim", {"entities": [(0, 11, "PER")]}),
    ("Elif Kaya", {"entities": [(0, 9, "PER")]}),
    ("Konstantin Ivanov", {"entities": [(0, 17, "PER")]}),
    ("Mohamed Salah", {"entities": [(0, 13, "PER")]}),
    ("Aisha Khan", {"entities": [(0, 10, "PER")]}),
    ("Emily Brown", {"entities": [(0, 11, "PER")]}),
    ("Hiroshi Sato", {"entities": [(0, 12, "PER")]}),
    ("Lena Petrova", {"entities": [(0, 12, "PER")]}),
    ("Marko Nikolic", {"entities": [(0, 13, "PER")]}),
    ("Olivia Smith", {"entities": [(0, 12, "PER")]}),
    ("Xia Liu", {"entities": [(0, 7, "PER")]}),
    ("Isabella Rossi", {"entities": [(0, 14, "PER")]}),
    ("Nicolas Garcia", {"entities": [(0, 14, "PER")]}),
    ("Fatima Mohammed", {"entities": [(0, 15, "PER")]}),
    ("Bartosz Kowalski", {"entities": [(0, 16, "PER")]}),
    ("Hassan Ali", {"entities": [(0, 10, "PER")]}),
    ("Tereza Novakova", {"entities": [(0, 15, "PER")]}),
    ("Ivan Petrov", {"entities": [(0, 10, "PER")]}),
    ("Ahmed Musa", {"entities": [(0, 10, "PER")]}),
    ("Anna Nowak", {"entities": [(0, 10, "PER")]}),
    ("Leonardo Silva", {"entities": [(0, 14, "PER")]}),
    ("Sophia Kim", {"entities": [(0, 10, "PER")]}),
    ("David Jones", {"entities": [(0, 11, "PER")]}),
    ("Rosa Lopez", {"entities": [(0, 10, "PER")]}),
    ("Ali Hassan", {"entities": [(0, 10, "PER")]}),
    ("Chen Li", {"entities": [(0, 7, "PER")]}),
    ("Elena Ivanova", {"entities": [(0, 13, "PER")]}),
    ("Lucas Oliveira", {"entities": [(0, 14, "PER")]}),
    ("Amira Abdallah", {"entities": [(0, 14, "PER")]}),
    ("Svenja Müller", {"entities": [(0, 12, "PER")]}),
    ("Kenji Takahashi", {"entities": [(0, 15, "PER")]}),
    ("Marta Fernandez", {"entities": [(0, 15, "PER")]}),
    ("Viktor Pavlov", {"entities": [(0, 13, "PER")]}),
    ("Jin Chen", {"entities": [(0, 8, "PER")]}),
    ("Klara Novak", {"entities": [(0, 11, "PER")]}),
    ("Hussein Mohamed", {"entities": [(0, 16, "PER")]}),
    ("David Nguyen", {"entities": [(0, 12, "PER")]}),
    ("Emilia Petrova", {"entities": [(0, 14, "PER")]}),
    ("Lars Johansen", {"entities": [(0, 13, "PER")]}),
    ("Sofia Santos", {"entities": [(0, 12, "PER")]}),
    ("Nina Müller", {"entities": [(0, 10, "PER")]}),
    ("Omar Ali", {"entities": [(0, 8, "PER")]}),
    ("Elif Demir", {"entities": [(0, 10, "PER")]}),
    ("Isabel Fernandez", {"entities": [(0, 15, "PER")]}),
    ("Lucas Rossi", {"entities": [(0, 11, "PER")]}),
    ("Hana Ali", {"entities": [(0, 8, "PER")]}),
    ("Kim Lee", {"entities": [(0, 7, "PER")]}),
    ("Rajesh Kumar", {"entities": [(0, 12, "PER")]}),
    ("Fatoumata Ba", {"entities": [(0, 12, "PER")]}),
    ("Nadia Schmidt", {"entities": [(0, 13, "PER")]}),
    ("Elias Zhang", {"entities": [(0, 11, "PER")]}),
    ("Lucia Neri", {"entities": [(0, 10, "PER")]}),
    ("Kwame Mensah", {"entities": [(0, 12, "PER")]}),
    ("Sara Sorensen", {"entities": [(0, 13, "PER")]}),
    ("Anastasia Ivanova", {"entities": [(0, 17, "PER")]}),
    ("Juan Morales", {"entities": [(0, 12, "PER")]}),
    ("Yuki Yamamoto", {"entities": [(0, 13, "PER")]}),
    ("Amina Abdi", {"entities": [(0, 10, "PER")]}),
    ("Marta Kowalczyk", {"entities": [(0, 15, "PER")]}),
    ("Mohammed Hassan", {"entities": [(0, 15, "PER")]}),
    ("Hannah Tanaka", {"entities": [(0, 13, "PER")]}),
    ("Viktor Pavlovic", {"entities": [(0, 15, "PER")]}),
    ("Fatima Ali", {"entities": [(0, 10, "PER")]}),
    ("Eli Cohen", {"entities": [(0, 9, "PER")]}),
    ("Sophia Suzuki", {"entities": [(0, 13, "PER")]}),
    ("Mehmet Yildiz", {"entities": [(0, 13, "PER")]}),
    ("Chloe Tran", {"entities": [(0, 10, "PER")]}),
    ("David Johnson", {"entities": [(0, 13, "PER")]}),
    ("Elena Petrovska", {"entities": [(0, 15, "PER")]}),
    ("Santiago Ramirez", {"entities": [(0, 16, "PER")]}),
    ("Nina Popescu", {"entities": [(0, 12, "PER")]}),
    ("Arjun Reddy", {"entities": [(0, 11, "PER")]}),
    ("Leila Bouzid", {"entities": [(0, 12, "PER")]}),
    ("Igor Petrov", {"entities": [(0, 11, "PER")]}),
    ("Isabel Pereira", {"entities": [(0, 14, "PER")]}),
    ("Jin Yang", {"entities": [(0, 7, "PER")]}),
    ("Olga Petrova", {"entities": [(0, 12, "PER")]}),
    ("Mateo Lopez", {"entities": [(0, 11, "PER")]}),
    ("Ming Zhao", {"entities": [(0, 9, "PER")]}),
    ("Aisha Ahmed", {"entities": [(0, 11, "PER")]}),
    ("Victoria Müller", {"entities": [(0, 15, "PER")]}),
    ("Fahad Rahman", {"entities": [(0, 12, "PER")]}),
    ("Emily White", {"entities": [(0, 11, "PER")]}),
    ("Sophia Hernandez", {"entities": [(0, 16, "PER")]}),
    ("Khalid Ibrahim", {"entities": [(0, 14, "PER")]}),
    ("Jana Novakova", {"entities": [(0, 13, "PER")]}),
    ("Ali Mahmoud", {"entities": [(0, 11, "PER")]}),
    ("Lucas Ferreira", {"entities": [(0, 14, "PER")]}),
    ("Alya Mohamed", {"entities": [(0, 12, "PER")]}),
    ("Julian Muller", {"entities": [(0, 13, "PER")]}),
    ("Daniela Santos", {"entities": [(0, 14, "PER")]}),
    ("Eva Andersen", {"entities": [(0, 12, "PER")]}),
    ("Jakub Novak", {"entities": [(0, 10, "PER")]}),
    ("Sofia Martins", {"entities": [(0, 13, "PER")]}),
    ("Musa Yusuf", {"entities": [(0, 10, "PER")]}),
    ("Elif Demirel", {"entities": [(0, 12, "PER")]}),
    ("Konstantin Petrov", {"entities": [(0, 17, "PER")]}),
    ("Mohamed El-Sayed", {"entities": [(0, 15, "PER")]}),
    ("Aisha Mohamed", {"entities": [(0, 13, "PER")]}),
    ("Emily Smith", {"entities": [(0, 11, "PER")]}),
    ("Hiroshi Yamamoto", {"entities": [(0, 15, "PER")]}),
    ("Lena Petrovska", {"entities": [(0, 13, "PER")]}),
    ("Marko Jovanovic", {"entities": [(0, 15, "PER")]}),
    ("Olivia Johnson", {"entities": [(0, 14, "PER")]}),
    ("Xia Chen", {"entities": [(0, 8, "PER")]}),
    ("Isabella Fernandes", {"entities": [(0, 17, "PER")]}),
    ("Nicolas Rodriguez", {"entities": [(0, 17, "PER")]}),
    ("Fatima Abdallah", {"entities": [(0, 15, "PER")]}),
    ("Bartosz Nowak", {"entities": [(0, 13, "PER")]}),
    ("Hassan Mohammed", {"entities": [(0, 15, "PER")]}),
    ("Tereza Novak", {"entities": [(0, 12, "PER")]}),
    ("Ivan Ivanov", {"entities": [(0, 10, "PER")]}),
    ("Ahmed Hassan", {"entities": [(0, 12, "PER")]}),
    ("Anna Kowalski", {"entities": [(0, 13, "PER")]}),
    ("Leonardo Santos", {"entities": [(0, 15, "PER")]}),
    ("Sophia Chen", {"entities": [(0, 10, "PER")]}),
    ("David Williams", {"entities": [(0, 14, "PER")]}),
    ("Rosa Garcia", {"entities": [(0, 10, "PER")]}),
    ("Ali Mohamed", {"entities": [(0, 11, "PER")]}),
    ("Chen Wu", {"entities": [(0, 7, "PER")]}),
    ("Elena Petrovska", {"entities": [(0, 15, "PER")]}),
    ("Lucas Fernandes", {"entities": [(0, 15, "PER")]}),
    ("Amira El-Sayed", {"entities": [(0, 14, "PER")]}),
    ("Svenja Schuster", {"entities": [(0, 15, "PER")]}),
    ("Kenji Yamamoto", {"entities": [(0, 14, "PER")]}),
    ("Marta Gonzalez", {"entities": [(0, 14, "PER")]}),
    ("Viktor Ivanov", {"entities": [(0, 13, "PER")]}),
    ("Jin Liu", {"entities": [(0, 7, "PER")]}),
    ("Klara Novakova", {"entities": [(0, 14, "PER")]}),
    ("Hussein El-Sayed", {"entities": [(0, 16, "PER")]}),
    ("David Tran", {"entities": [(0, 10, "PER")]}),
    ("Emilia Petrovska", {"entities": [(0, 16, "PER")]}),
    ("Lars Johansen", {"entities": [(0, 13, "PER")]}),
    ("Sofia Oliveira", {"entities": [(0, 14, "PER")]}),
    ("Nina Muller", {"entities": [(0, 10, "PER")]}),
    ("Omar Hussein", {"entities": [(0, 12, "PER")]}),
    ("Elif Kaya", {"entities": [(0, 9, "PER")]}),
    ("Isabel Oliveira", {"entities": [(0, 15, "PER")]}),
    ("Abdul Rahman", {"entities": [(0, 12, "PER")]}),
    ("Lucia Rodriguez", {"entities": [(0, 15, "PER")]}),
    ("Manuel Pereira", {"entities": [(0, 14, "PER")]}),
    ("Amira Khaled", {"entities": [(0, 12, "PER")]}),
    ("Sven Möller", {"entities": [(0, 10, "PER")]}),
    ("Kim Nguyen", {"entities": [(0, 9, "PER")]}),
    ("Alejandro Ramirez", {"entities": [(0, 16, "PER")]}),
    ("Maria Fernandez", {"entities": [(0, 14, "PER")]}),
    ("Jose Martinez", {"entities": [(0, 13, "PER")]}),
    ("Anya Ivanova", {"entities": [(0, 11, "PER")]}),
    ("Hasan Ahmed", {"entities": [(0, 11, "PER")]}),
    ("Sofia Dimitrova", {"entities": [(0, 15, "PER")]}),
    ("Martin Schmidt", {"entities": [(0, 14, "PER")]}),
    ("Daniela Marino", {"entities": [(0, 14, "PER")]}),
    ("Salim Hassan", {"entities": [(0, 12, "PER")]}),
    ("Sofia Dimitrova", {"entities": [(0, 15, "PER")]}),
    ("Fahad Hassan", {"entities": [(0, 12, "PER")]}),
    ("Emilia Fischer", {"entities": [(0, 14, "PER")]}),
    ("Anna Rossi", {"entities": [(0, 10, "PER")]}),
    ("Viktor Popov", {"entities": [(0, 12, "PER")]}),
    ("Julia Wagner", {"entities": [(0, 12, "PER")]}),
    ("Elena Petrova", {"entities": [(0, 13, "PER")]}),
    ("Diego Morales", {"entities": [(0, 13, "PER")]}),
    ("Ming Zhao", {"entities": [(0, 9, "PER")]}),
    ("Marta Kowalski", {"entities": [(0, 14, "PER")]}),
    ("Ali Hassan", {"entities": [(0, 10, "PER")]}),
    ("Nadia Schmidt", {"entities": [(0, 13, "PER")]}),
    ("David Kim", {"entities": [(0, 9, "PER")]}),
    ("Olga Ivanov", {"entities": [(0, 11, "PER")]}),
    ("Chen Wei", {"entities": [(0, 8, "PER")]}),
    ("Carlos Silva", {"entities": [(0, 12, "PER")]}),
    ("Sofia Gomez", {"entities": [(0, 11, "PER")]}),
    ("Ivan Petrov", {"entities": [(0, 10, "PER")]}),
    ("Sara Svensson", {"entities": [(0, 13, "PER")]}),
    ("David Smith", {"entities": [(0, 11, "PER")]}),
    ("Fatima Mohammed", {"entities": [(0, 15, "PER")]}),
    ("Elias Cohen", {"entities": [(0, 11, "PER")]}),
    ("Emily Johnson", {"entities": [(0, 13, "PER")]}),
    ("Anastasia Ivanov", {"entities": [(0, 16, "PER")]}),
    ("Juan Martinez", {"entities": [(0, 12, "PER")]}),
    ("Mehmet Yilmaz", {"entities": [(0, 13, "PER")]}),
    ("Chloe Nguyen", {"entities": [(0, 11, "PER")]}),
    ("Yuki Nakamura", {"entities": [(0, 13, "PER")]}),
    ("Daniela Silva", {"entities": [(0, 13, "PER")]}),
    ("Leila Ben Ali", {"entities": [(0, 13, "PER")]}),
    ("Lucas Martins", {"entities": [(0, 13, "PER")]}),
    ("Amina Diallo", {"entities": [(0, 12, "PER")]}),
    ("Maria Garcia", {"entities": [(0, 12, "PER")]}),
    ("Giulia Rossi", {"entities": [(0, 12, "PER")]}),
    ("Ahmed Khan", {"entities": [(0, 10, "PER")]}),
    ("Chen Wei", {"entities": [(0, 8, "PER")]}),
    ("Ali Rahman", {"entities": [(0, 10, "PER")]}),
    ("Hannah Lee", {"entities": [(0, 10, "PER")]}),
    ("Santiago Lopez", {"entities": [(0, 14, "PER")]}),
    ("Olga Ivanova", {"entities": [(0, 12, "PER")]}),
    ("Jin Park", {"entities": [(0, 8, "PER")]}),
    ("Mateo Garcia", {"entities": [(0, 12, "PER")]}),
    ("Lucas Oliveira", {"entities": [(0, 14, "PER")]}),
    ("Omar Ali", {"entities": [(0, 8, "PER")]}),
    ("Fatima Hassan", {"entities": [(0, 13, "PER")]}),
    ("Sara Svensson", {"entities": [(0, 13, "PER")]}),
    ("Yuki Nakamura", {"entities": [(0, 13, "PER")]}),
    ("Aisha Abdi", {"entities": [(0, 10, "PER")]}),
    ("Marta Kowalski", {"entities": [(0, 14, "PER")]}),
    ("Mohammed Ali", {"entities": [(0, 12, "PER")]}),
    ("Hannah Lee", {"entities": [(0, 10, "PER")]}),
    ("Viktor Petrov", {"entities": [(0, 13, "PER")]}),
    ("Elias Cohen", {"entities": [(0, 11, "PER")]}),
    ("Sophia Tanaka", {"entities": [(0, 13, "PER")]}),
    ("Mehmet Yilmaz", {"entities": [(0, 13, "PER")]}),
    ("Chloe Nguyen", {"entities": [(0, 11, "PER")]}),
    ("David Smith", {"entities": [(0, 11, "PER")]}),
    ("Elena Petrova", {"entities": [(0, 13, "PER")]}),
    ("Nina Popescu", {"entities": [(0, 12, "PER")]}),
    ("Arjun Patel", {"entities": [(0, 11, "PER")]}),
    ("Igor Petrovic", {"entities": [(0, 13, "PER")]}),
    ("Isabel Costa", {"entities": [(0, 12, "PER")]}),
    ("Olga Ivanova", {"entities": [(0, 12, "PER")]}),
    ("Mateo Garcia", {"entities": [(0, 12, "PER")]}),
    ("Ming Li", {"entities": [(0, 7, "PER")]}),
    ("Viktoria Muller", {"entities": [(0, 15, "PER")]}),
    ("Fahad Khan", {"entities": [(0, 10, "PER")]}),
    ("Emily Johnson", {"entities": [(0, 13, "PER")]}),
    ("Sophia Rodriguez", {"entities": [(0, 15, "PER")]}),
    ("Jana Novak", {"entities": [(0, 10, "PER")]}),
    ("Lucas Martins", {"entities": [(0, 13, "PER")]}),
    ("Julian Schmidt", {"entities": [(0, 14, "PER")]}),
    ("Daniela Silva", {"entities": [(0, 13, "PER")]}),
    ("Jakub Nowak", {"entities": [(0, 11, "PER")]}),
    ("Sofia Garcia", {"entities": [(0, 12, "PER")]}),
    ("Elif Kaya", {"entities": [(0, 9, "PER")]}),
    ("Konstantin Ivanov", {"entities": [(0, 17, "PER")]}),
    ("Mohamed Salah", {"entities": [(0, 13, "PER")]}),
    ("Aisha Khan", {"entities": [(0, 10, "PER")]}),
    ("Emily Brown", {"entities": [(0, 11, "PER")]}),
    ("Hiroshi Sato", {"entities": [(0, 12, "PER")]}),
    ("Lena Petrova", {"entities": [(0, 12, "PER")]}),
    ("Marko Nikolic", {"entities": [(0, 13, "PER")]}),
    ("Olivia Smith", {"entities": [(0, 12, "PER")]}),
    ("Isabella Rossi", {"entities": [(0, 14, "PER")]}),
    ("Fatima Mohammed", {"entities": [(0, 15, "PER")]}),
    ("Bartosz Kowalski", {"entities": [(0, 16, "PER")]}),
    ("Tereza Novakova", {"entities": [(0, 15, "PER")]}),
    ("Ivan Petrov", {"entities": [(0, 10, "PER")]}),
    ("Ahmed Musa", {"entities": [(0, 10, "PER")]}),
    ("Anna Nowak", {"entities": [(0, 10, "PER")]}),
    ("Leonardo Silva", {"entities": [(0, 14, "PER")]}),
    ("Sophia Kim", {"entities": [(0, 10, "PER")]}),
    ("David Jones", {"entities": [(0, 11, "PER")]}),
    ("Rosa Lopez", {"entities": [(0, 10, "PER")]}),
    ("Chen Li", {"entities": [(0, 7, "PER")]}),
    ("Elena Ivanova", {"entities": [(0, 13, "PER")]}),
    ("Lucas Oliveira", {"entities": [(0, 14, "PER")]}),
    ("Amira Abdallah", {"entities": [(0, 14, "PER")]}),
    ("Kenji Takahashi", {"entities": [(0, 15, "PER")]}),
    ("Marta Fernandez", {"entities": [(0, 15, "PER")]}),
    ("Jin Chen", {"entities": [(0, 8, "PER")]}),
    ("Klara Novak", {"entities": [(0, 11, "PER")]}),
    ("David Nguyen", {"entities": [(0, 12, "PER")]}),
    ("Emilia Petrova", {"entities": [(0, 14, "PER")]}),
    ("Lars Johansen", {"entities": [(0, 13, "PER")]}),
    ("Sofia Santos", {"entities": [(0, 12, "PER")]}),
    ("Nina Müller", {"entities": [(0, 10, "PER")]}),
    ("Elif Demir", {"entities": [(0, 10, "PER")]}),
    ("Isabel Fernandez", {"entities": [(0, 15, "PER")]}),
    ("Lucas Rossi", {"entities": [(0, 11, "PER")]}),
    ("Hana Ali", {"entities": [(0, 8, "PER")]}),
    ("Kim Lee", {"entities": [(0, 7, "PER")]}),
    ("Rajesh Kumar", {"entities": [(0, 12, "PER")]}),
    ("Fatoumata Ba", {"entities": [(0, 12, "PER")]}),
    ("Nadia Schmidt", {"entities": [(0, 13, "PER")]}),
    ("Elias Zhang", {"entities": [(0, 11, "PER")]}),
    ("Lucia Neri", {"entities": [(0, 10, "PER")]}),
    ("Kwame Mensah", {"entities": [(0, 12, "PER")]}),
    ("Sara Sorensen", {"entities": [(0, 13, "PER")]}),
    ("Anastasia Ivanova", {"entities": [(0, 17, "PER")]}),
    ("Juan Morales", {"entities": [(0, 12, "PER")]}),
    ("Yuki Yamamoto", {"entities": [(0, 13, "PER")]}),
    ("Amina Abdi", {"entities": [(0, 10, "PER")]}),
    ("Marta Kowalczyk", {"entities": [(0, 15, "PER")]}),
    ("Mohammed Hassan", {"entities": [(0, 15, "PER")]}),
    ("Hannah Tanaka", {"entities": [(0, 13, "PER")]}),
    ("Fatima Ali", {"entities": [(0, 10, "PER")]}),
    ("Eli Cohen", {"entities": [(0, 9, "PER")]}),
    ("Sophia Suzuki", {"entities": [(0, 13, "PER")]}),
    ("Chloe Tran", {"entities": [(0, 10, "PER")]}),
    ("David Johnson", {"entities": [(0, 13, "PER")]}),
    ("Elena Petrovska", {"entities": [(0, 15, "PER")]}),
    ("Nina Popescu", {"entities": [(0, 12, "PER")]}),
    ("Arjun Reddy", {"entities": [(0, 11, "PER")]}),
    ("Leila Bouzid", {"entities": [(0, 12, "PER")]}),
    ("Igor Petrov", {"entities": [(0, 11, "PER")]}),
    ("Isabel Pereira", {"entities": [(0, 14, "PER")]}),
    ("Jin Yang", {"entities": [(0, 7, "PER")]}),
    ("Olga Petrova", {"entities": [(0, 12, "PER")]}),
    ("Mateo Lopez", {"entities": [(0, 11, "PER")]}),
    ("Ming Zhao", {"entities": [(0, 9, "PER")]}),
    ("Aisha Ahmed", {"entities": [(0, 11, "PER")]}),
    ("Victoria Müller", {"entities": [(0, 15, "PER")]}),
    ("Fahad Rahman", {"entities": [(0, 12, "PER")]}),
    ("Emily White", {"entities": [(0, 11, "PER")]}),
    ("Sophia Hernandez", {"entities": [(0, 16, "PER")]}),
    ("Khalid Ibrahim", {"entities": [(0, 14, "PER")]}),
    ("Jana Novakova", {"entities": [(0, 13, "PER")]}),
    ("Ali Mahmoud", {"entities": [(0, 11, "PER")]}),
    ("Lucas Ferreira", {"entities": [(0, 14, "PER")]}),
    ("Alya Mohamed", {"entities": [(0, 12, "PER")]}),
    ("Julian Muller", {"entities": [(0, 13, "PER")]}),
    ("Daniela Santos", {"entities": [(0, 14, "PER")]}),
    ("Eva Andersen", {"entities": [(0, 12, "PER")]}),
    ("Jakub Novak", {"entities": [(0, 10, "PER")]}),
    ("Sofia Martins", {"entities": [(0, 13, "PER")]}),
    ("Musa Yusuf", {"entities": [(0, 10, "PER")]}),
    ("Elif Demirel", {"entities": [(0, 12, "PER")]}),
    ("Konstantin Petrov", {"entities": [(0, 17, "PER")]}),
    ("Mohamed El-Sayed", {"entities": [(0, 15, "PER")]}),
    ("Aisha Mohamed", {"entities": [(0, 13, "PER")]}),
    ("Emily Smith", {"entities": [(0, 11, "PER")]}),
    ("Hiroshi Yamamoto", {"entities": [(0, 15, "PER")]}),
    ("Lena Petrovska", {"entities": [(0, 13, "PER")]}),
    ("Marko Jovanovic", {"entities": [(0, 15, "PER")]}),
    ("Olivia Johnson", {"entities": [(0, 14, "PER")]}),
    ("Xia Chen", {"entities": [(0, 8, "PER")]}),
    ("Isabella Fernandes", {"entities": [(0, 17, "PER")]}),
    ("Nicolas Rodriguez", {"entities": [(0, 17, "PER")]}),
    ("Fatima Abdallah", {"entities": [(0, 15, "PER")]}),
    ("Bartosz Nowak", {"entities": [(0, 13, "PER")]}),
    ("Hassan Mohammed", {"entities": [(0, 15, "PER")]}),
    ("Tereza Novak", {"entities": [(0, 12, "PER")]}),
    ("Ivan Ivanov", {"entities": [(0, 10, "PER")]}),
    ("Ahmed Hassan", {"entities": [(0, 12, "PER")]}),
    ("Anna Kowalski", {"entities": [(0, 13, "PER")]}),
    ("Leonardo Santos", {"entities": [(0, 15, "PER")]}),
    ("Sophia Chen", {"entities": [(0, 10, "PER")]}),
    ("David Williams", {"entities": [(0, 14, "PER")]}),
    ("Rosa Garcia", {"entities": [(0, 10, "PER")]}),
    ("Ali Mohamed", {"entities": [(0, 11, "PER")]}),
    ("Chen Wu", {"entities": [(0, 7, "PER")]}),
    ("Elena Petrovska", {"entities": [(0, 15, "PER")]}),
    ("Lucas Fernandes", {"entities": [(0, 15, "PER")]}),
    ("Amira El-Sayed", {"entities": [(0, 14, "PER")]}),
    ("Svenja Schuster", {"entities": [(0, 15, "PER")]}),
    ("Kenji Yamamoto", {"entities": [(0, 14, "PER")]}),
    ("Marta Gonzalez", {"entities": [(0, 14, "PER")]}),
    ("Viktor Ivanov", {"entities": [(0, 13, "PER")]}),
    ("Jin Liu", {"entities": [(0, 7, "PER")]}),
    ("Klara Novakova", {"entities": [(0, 14, "PER")]}),
    ("Hussein El-Sayed", {"entities": [(0, 16, "PER")]}),
    ("David Tran", {"entities": [(0, 10, "PER")]}),
    ("Emilia Petrovska", {"entities": [(0, 16, "PER")]}),
    ("Lars Johansen", {"entities": [(0, 13, "PER")]}),
    ("Sofia Oliveira", {"entities": [(0, 14, "PER")]}),
    ("Nina Muller", {"entities": [(0, 10, "PER")]}),
    ("Omar Hussein", {"entities": [(0, 12, "PER")]}),
    ("Elif Kaya", {"entities": [(0, 9, "PER")]}),
    ("Isabel Oliveira", {"entities": [(0, 15, "PER")]}),

 ]

    full_sentence_training_data = [
        ("Impressum Rechtsanwaltskanzlei Schmidt, RA Dr. Hans Mustermann, Email: kanzlei@anwalt-paderborn.de",
         {"entities": [(10, 36, "ORG"), (38, 56, "PER"), (65, 91, "EMAIL")]}),
        ("Kontakt Anwaltsbüro Müller & Partner, Frau Erika Musterfrau, E-Mail: kontakt@rae-strake.de",
         {"entities": [(8, 35, "ORG"), (37, 53, "PER"), (62, 83, "EMAIL")]}),
        ("Rechtsanwälte Fischer & Söhne GmbH, Herr Max Mustermann, Email: info@eikel-partner.de",
         {"entities": [(0, 32, "ORG"), (34, 51, "PER"), (60, 81, "EMAIL")]}),
        ("Kanzlei Mustermann & Partner Rechtsanwälte, Dr. Michael Bauer, E-Mail: ashkan@ra-ashkan.de",
         {"entities": [(0, 41, "ORG"), (43, 59, "PER"), (68, 86, "EMAIL")]}),
        ("Rechtsanwalt Dr. Karl Schmidt, Kontakt: info@rae-schaefers.de",
         {"entities": [(0, 25, "PER"), (35, 57, "EMAIL")]}),
        ("Notarin Dr. Anna Berger, E-Mail: zentrale@kanzlei-am-rosentor.de",
         {"entities": [(8, 24, "PER"), (34, 66, "EMAIL")]}),
        ("Rechtsanwältin Julia König, Email: info@steinertstrafrecht.com",
         {"entities": [(0, 25, "PER"), (33, 60, "EMAIL")]}),
        ("Anwaltskanzlei Hansen & Partner mbB, Dr. Andrea Schulze, LL.M., Email: kanzlei@anwalt-paderborn.com",
         {"entities": [(0, 33, "ORG"), (35, 59, "PER"), (68, 97, "EMAIL")]}),
        ("Impressum: Anwaltskanzlei Maier & Kollegen, Kontakt: kanzlei@anwalt-paderborn.de",
         {"entities": [(11, 39, "ORG"), (49, 75, "EMAIL")]}),
        ("Kanzlei Müller und Partner mbB, Email: info@eikel-partner.de",
         {"entities": [(0, 31, "ORG"), (39, 60, "EMAIL")]}),
    ]

    training_data = isolated_training_data + full_sentence_training_data

    scraper.train_nlp_model(training_data)

    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(scraper.extract_info_from_url, url): url for url in urls}
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                info = future.result()
                if info:
                    results.append(info)
                    print(f"Processed: {info}")
                    print(f"Confidence: {info.confidence}")
            except Exception as exc:
                print(f"{url} generated an exception: {exc}")

    results.sort(key=lambda x: sum(x.confidence.values()) / len(x.confidence), reverse=True)

    with open('results.json', 'w', encoding='utf-8') as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)

    return results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    urls = [
        "https://www.diedigitalekanzlei.com",
        "https://www.anwalt-paderborn.de",
        "https://www.le-mans-wall.de",
        "https://www.rae-strake.de",
        "https://eikel-partner.de",
        "https://www.anwalt-paderborn.com",
        "https://rae-schaefers.de",
        "https://kanzlei-am-rosentor.de",
        "https://www.warm-rechtsanwaelte.de",
        "https://www.rehmann.de",
        "https://ashkan-rechtsanwalt-arbeitsrecht-paderborn.de",
        "https://steinertstrafrecht.com"
    ]

    max_workers = min(5, len(urls))
    results = main(urls, max_workers=max_workers)
    print(f"Total processed URLs: {len(results)}")

    print("\nTop 3 results:")
    for result in results[:3]:
        print(f"URL: {result.url}")
        print(f"Company: {result.company_name}")
        print(f"Contact: {result.contact_name}")
        print(f"Email: {result.email}")
        print(f"Average Confidence: {sum(result.confidence.values()) / len(result.confidence):.2f}")
        print()

    print(f"Full results exported to results.json")
