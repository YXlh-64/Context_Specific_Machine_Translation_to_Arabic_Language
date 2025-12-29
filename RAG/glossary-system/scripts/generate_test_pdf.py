from fpdf import FPDF
import os

def create_media_test_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    title = "Challenges in Modern Media Production and Ethics"
    
    # CORRECTED TEXT: Switched "worm’s" to "worm's" (straight apostrophe)
    text = (
        "The recent controversy at the domestic news agency began when the board of directors "
        "voted to block a feature article feature column that was intended to splash a piece of news "
        "regarding government censorship. This decision sparked an intense debate about freedom of the press "
        "and the fundamental right to express and disseminate opinions. Critics argued that the "
        "board of editors at the daily morning paper had bowed to pressure, failing their duty to inform "
        "the public. However, management insisted that the dissemination of information must always be "
        "balanced against the protection of law and order, especially when sensitive national security "
        "matters are involved.\n\n"
        
        "On the technical side of the production, the director of photography faced significant hurdles "
        "while finalizing the documentary, which is currently a work in the process of publication. "
        "For artistic reasons, the team decided to shoot on black and white film. The opening sequence "
        "features a stunning distant shot master shot of the city skyline, which immediately cuts to a "
        "dramatic close up shot of the protagonist. To achieve a unique perspective, the camera crew "
        "also utilized a worm's eye view shot during the protest scenes. Despite these artistic "
        "achievements, the tape to tape editing process has been slow due to technical glitches in the "
        "video head record and playback system.\n\n"
        
        "Legal issues have further complicated the release. The standards and practices continuity "
        "department warned that certain scenes might constitute an offence against public morals "
        "if broadcast via direct broadcasting satellites without editing. Consequently, the "
        "board of control has requested a review before the film is distributed on a digital versatile disk "
        "or released for community antenna television. There is also a concern that the extensive use of "
        "archival footage might infringe on a work out of copyright, although the producers claim "
        "all rights reserved under current laws. Ultimately, the goal is to ensure a free flow of information "
        "without violating the privacy of the man on the street."
    )

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, title, ln=True, align='C')
    pdf.ln(10)
    
    # Body
    # .encode('latin-1', 'replace').decode('latin-1') ensures any other hidden special chars are handled
    clean_text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, clean_text)
    
    filename = "media_test_doc.pdf"
    pdf.output(filename)
    print(f"PDF created successfully: {os.path.abspath(filename)}")

if __name__ == "__main__":
    create_media_test_pdf()