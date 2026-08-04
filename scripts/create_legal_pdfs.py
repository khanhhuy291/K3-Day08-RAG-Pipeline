"""
Script tạo 3 file PDF chính sách đại học cho Task 1.
Sử dụng fpdf2 (FPDF2 class) để tạo PDF với nội dung thực tế về RMIT Vietnam.
"""
from pathlib import Path
from fpdf import FPDF


OUTPUT_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def make_pdf(title: str, sections: list[tuple[str, list[str]]]) -> FPDF:
    """Create a simple PDF with title and sections of bullet lines."""
    pdf = FPDF()
    # IMPORTANT: set_margins must be called BEFORE add_page
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "I", 9)
    pdf.cell(0, 6, "RMIT University Vietnam | Academic Year 2025-2026", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(6)

    for heading, lines in sections:
        # Section heading
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, heading, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

        # Body lines — use write() which handles long text gracefully
        pdf.set_font("Helvetica", "", 10)
        for line in lines:
            pdf.write(6, f"  - {line}")
            pdf.ln(6)
        pdf.ln(3)

    return pdf


# ─────────────────────────────────────────────────────────────────────────────
# 1. Tuition Fees Policy
# ─────────────────────────────────────────────────────────────────────────────
def create_tuition_fees_pdf():
    sections = [
        ("1. Overview", [
            "RMIT University Vietnam charges fees based on program, level, and campus.",
            "All fees are quoted in Vietnamese Dong (VND) and reviewed annually.",
            "30-day notice required before any fee adjustments.",
        ]),
        ("2. Undergraduate Program Fees (per semester, 15 credit points)", [
            "Business Administration (Bachelor): VND 38,500,000",
            "Information Technology (Bachelor): VND 41,200,000",
            "Engineering (Bachelor): VND 43,800,000",
            "Communication Design (Bachelor): VND 39,600,000",
            "Accounting (Bachelor): VND 37,900,000",
        ]),
        ("3. Postgraduate Program Fees", [
            "Master of Business Administration (MBA): VND 158,000,000 total",
            "Master of Information Technology: VND 143,500,000 total",
            "Graduate Certificate Programs: VND 58,000,000 to VND 75,000,000",
        ]),
        ("4. Payment Schedule & Deadlines", [
            "Semester 1 (February to June): Payment due 7 February 2026",
            "Semester 2 (July to November): Payment due 6 July 2026",
            "Semester 3 (November to February): Payment due 2 November 2026",
            "Instalment plans available - apply 14 days before deadline.",
        ]),
        ("5. Accepted Payment Methods", [
            "Bank Transfer to RMIT Vietnam designated VND accounts",
            "Online via myRMIT Student Portal (VISA/Mastercard accepted)",
            "Direct payment at Student Finance Office (cash or bank cheque)",
            "VNPay QR code payment at Student Finance counters",
            "Momo e-wallet for amounts under VND 50,000,000 per transaction",
        ]),
        ("6. Late Payment Penalties", [
            "Late fee: VND 500,000 per week of delay after due date",
            "System lock after 3 weeks: No access to Canvas LMS or myRMIT",
            "Enrolment suspension after 6 weeks of non-payment",
            "Academic transcript withheld until outstanding fees are settled",
        ]),
        ("7. Refund Policy", [
            "Before census date (Week 2): 100% refund of subject fee",
            "Week 3-4: 50% partial refund",
            "Week 5 onwards: No refund applicable",
            "Medical withdrawal: Full refund subject to documentation",
        ]),
        ("8. Financial Hardship Assistance", [
            "Emergency bursary available via Student Financial Assistance Program",
            "Apply with supporting documents 10 business days before deadline",
            "Contact: studentfinance@rmit.edu.vn or +84 28 3776 1300",
        ]),
    ]
    pdf = make_pdf("Tuition Fees & Payment Policy", sections)
    out = OUTPUT_DIR / "tuition-fees-rmit.pdf"
    pdf.output(str(out))
    print(f"  [OK] {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Scholarship Policy
# ─────────────────────────────────────────────────────────────────────────────
def create_scholarship_pdf():
    sections = [
        ("1. Overview", [
            "RMIT Vietnam offers merit-based and need-based scholarships.",
            "All awards are tuition fee waivers unless otherwise stated.",
            "Applications managed through the myRMIT Student Portal.",
        ]),
        ("2. Academic Achievement Scholarship", [
            "Award: 50% tuition fee waiver per semester",
            "Eligibility: GPA >= 3.5 (out of 4.0) in preceding semester",
            "Minimum enrolment: 12 credit points per semester",
            "Renewable each semester subject to GPA maintenance",
            "Automatically assessed - no application required",
        ]),
        ("3. RMIT Excellence Scholarship (New Students)", [
            "Award: 100% tuition waiver for Year 1 (2 semesters)",
            "Eligibility: National HSE score >= 27/30 OR IELTS >= 7.5",
            "Application deadline: 30 November each year",
            "Requires transcripts and personal statement (500 words)",
            "Shortlisted candidates may be called for an interview",
        ]),
        ("4. Industry Partnership Scholarship", [
            "Sponsors: Unilever, Intel, FPT, HSBC, Shopee and others",
            "Award: 30% to 100% tuition waiver depending on sponsor",
            "Additional: Internship placement and mentoring program",
            "Application: Separate submissions required per sponsor",
        ]),
        ("5. Need-Based Bursary Program", [
            "Award: VND 15,000,000 to VND 40,000,000 per academic year (non-repayable)",
            "Eligibility: Annual household income below VND 120,000,000",
            "Requires income declaration and family verification",
            "Rolling applications accepted throughout the year",
        ]),
        ("6. Application Process", [
            "Step 1: Log in to myRMIT -> Scholarships & Funding -> Apply Now",
            "Step 2: Complete the online application form",
            "Step 3: Upload required documents (transcripts, income proof)",
            "Step 4: Submit before published deadline",
            "Step 5: Notification via student email within 4-6 weeks",
        ]),
        ("7. Conditions & Obligations", [
            "Maintain minimum GPA specified for the award",
            "Must be enrolled full-time (min 12 credit points per semester)",
            "Scholarships cannot be transferred, deferred, or converted to cash",
            "Attend annual RMIT Scholarship Ceremony (mandatory)",
            "Academic misconduct may result in immediate revocation",
        ]),
        ("8. Contact", [
            "Email: scholarships@rmit.edu.vn",
            "Phone: +84 28 3776 1300 (ext. 300)",
            "Walk-in: Student Connect, Building A, Level 1, SGS Campus",
            "Online chat available weekdays 08:00-17:00 on RMIT website",
        ]),
    ]
    pdf = make_pdf("Scholarship & Financial Aid Policy", sections)
    out = OUTPUT_DIR / "scholarship-policy-rmit.pdf"
    pdf.output(str(out))
    print(f"  [OK] {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Accommodation Services Policy
# ─────────────────────────────────────────────────────────────────────────────
def create_accommodation_pdf():
    sections = [
        ("1. Overview", [
            "RMIT Vietnam operates student residences at SGS (HCMC) and Hanoi campuses.",
            "Facilities managed by the RMIT Residential Services Division.",
            "Safe, supportive living environment for enrolled students.",
        ]),
        ("2. SGS Campus Residence Rates (per month, utilities included)", [
            "Standard Single Room (18m2): VND 4,800,000/month",
            "Deluxe Single Room (22m2): VND 6,200,000/month",
            "Twin Share Room (28m2): VND 3,500,000/person/month",
            "Amenities: Air conditioning, 100Mbps WiFi, en-suite bathroom",
            "Total capacity: 480 rooms across 3 residential towers",
        ]),
        ("3. Hanoi Campus Residence Rates (per month, utilities included)", [
            "Standard Single Room (20m2): VND 5,100,000/month",
            "Twin Share Room (30m2): VND 3,800,000/person/month",
            "Amenities: Air conditioning, WiFi, study desks, common kitchen",
            "Total capacity: 220 rooms in 2 residential blocks",
        ]),
        ("4. Eligibility & Priority", [
            "Priority 1: International students (non-Vietnamese passport holders)",
            "Priority 2: Domestic students from provinces other than HCMC or Hanoi",
            "Priority 3: Students with documented disabilities",
            "Priority 4: Continuing students with GPA >= 3.0",
            "Priority 5: General applications (first-come first-served)",
        ]),
        ("5. Application Process", [
            "Applications open 1 November each year for following academic year",
            "Apply via: myRMIT Portal -> Housing & Accommodation -> Apply",
            "Required: Student ID, passport/ID, emergency contact details",
            "Refundable deposit: VND 2,000,000 (returned upon check-out)",
            "Room offer confirmation within 10 business days",
            "Acceptance deadline: 5 business days from offer date",
        ]),
        ("6. Residence Rules", [
            "Quiet hours: 22:00 - 07:00 (Sun-Thu); 00:00 - 08:00 (Fri-Sat)",
            "No smoking on any residential premises (designated areas available)",
            "No pets of any kind permitted",
            "Guests allowed 08:00-22:00 daily; overnight guests not permitted",
            "Cooking only in designated kitchen areas",
            "Room inspections with 24-hour advance notice",
        ]),
        ("7. Support Services", [
            "24/7 Reception & Security desk in each residential building",
            "Resident Advisors (RAs) available for peer support",
            "Maintenance requests: myRMIT portal or +84 28 3776 1300 (ext. 200)",
            "Common facilities: Gym, study rooms, laundry, rooftop terrace",
        ]),
        ("8. Checkout & Lease Termination", [
            "Standard lease: One full semester (approx. 5 months)",
            "Early termination: 30 days written notice required",
            "Early termination fee: 1 month rent if notice < 30 days",
            "Security deposit returned within 14 business days of check-out",
        ]),
    ]
    pdf = make_pdf("Student Accommodation Services Policy", sections)
    out = OUTPUT_DIR / "accommodation-services-rmit.pdf"
    pdf.output(str(out))
    print(f"  [OK] {out}")


if __name__ == "__main__":
    print("Creating legal PDF documents for Task 1...")
    create_tuition_fees_pdf()
    create_scholarship_pdf()
    create_accommodation_pdf()
    print(f"\nDone! Files saved to: {OUTPUT_DIR}")
    pdfs = list(OUTPUT_DIR.glob("*.pdf"))
    print(f"Total PDF files: {len(pdfs)}")
    for p in pdfs:
        print(f"  - {p.name} ({p.stat().st_size:,} bytes)")
