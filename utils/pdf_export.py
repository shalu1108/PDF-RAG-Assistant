from fpdf import FPDF

def save_summary_pdf(filename, text):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)

    # Built-in fonts only support latin-1 characters, so strip anything else
    safe_text = text.encode("latin-1", "ignore").decode("latin-1")

    pdf.multi_cell(0, 8, safe_text)
    pdf.output(filename)