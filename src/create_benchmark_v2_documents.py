
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors


OUTPUT_DIR = Path("data/benchmark_v2/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DOCUMENTS = {
    "Apex_Compensation_Policy.pdf": {
        "title": "Apex Technologies - Compensation and Benefits Policy",
        "sections": [
            ("1. Salary Payment",
             "Employees receive their salaries on the last working day of each month. "
             "Salary payments are processed through the company's designated payroll system."),
            ("2. Annual Performance Bonus",
             "Eligible employees may receive an annual performance bonus of up to 15% "
             "of their base salary, depending on individual and company performance."),
            ("3. Retirement Benefits",
             "The company contributes 8% of an eligible employee's basic salary "
             "to the employee retirement fund each month."),
            ("4. Health Benefits",
             "Full-time employees become eligible for health insurance after completing "
             "three months of continuous employment. The policy covers eligible medical "
             "expenses subject to the insurer's terms."),
            ("5. Professional Development",
             "Employees may claim reimbursement of up to INR 25,000 per financial year "
             "for approved professional courses and certifications."),
        ],
    },
    "Apex_IT_Security_Policy.pdf": {
        "title": "Apex Technologies - IT and Information Security Policy",
        "sections": [
            ("1. Account Security",
             "Employees must use passwords containing at least 14 characters. "
             "Passwords must include uppercase letters, lowercase letters, numbers, "
             "and special characters."),
            ("2. Authentication",
             "Multi-factor authentication is mandatory for all corporate email accounts, "
             "cloud services, and remote access connections."),
            ("3. Device Management",
             "Company laptops must use full-disk encryption. Employees must install "
             "security updates within 7 calendar days of their release."),
            ("4. Data Classification",
             "Confidential information may only be accessed by employees with explicit "
             "business authorization. Public information does not require special access approval."),
            ("5. Incident Reporting",
             "Employees must report suspected cybersecurity incidents to the IT security "
             "team within 30 minutes of discovery."),
        ],
    },
    "Apex_Workplace_Policy.pdf": {
        "title": "Apex Technologies - Workplace and Attendance Policy",
        "sections": [
            ("1. Working Hours",
             "The standard work schedule is 10:00 AM to 7:00 PM, Monday through Friday."),
            ("2. Remote Work",
             "Employees may work remotely up to 3 days per week with written manager approval."),
            ("3. Probation",
             "New employees complete a probation period of 4 months from their joining date."),
            ("4. Annual Leave",
             "Eligible full-time employees receive 18 days of annual leave per calendar year."),
            ("5. Attendance",
             "Employees arriving more than 10 minutes after their scheduled start time "
             "must notify their reporting manager."),
        ],
    },
    "Apex_Expense_Policy.pdf": {
        "title": "Apex Technologies - Travel and Expense Reimbursement Policy",
        "sections": [
            ("1. Business Travel",
             "Employees may claim domestic business travel expenses up to INR 6,000 per day, "
             "subject to prior manager approval."),
            ("2. Meal Reimbursement",
             "The maximum reimbursable meal expense is INR 800 per employee per day."),
            ("3. Internet Reimbursement",
             "Eligible remote employees may claim up to INR 1,200 per month for home internet expenses."),
            ("4. Local Transport",
             "Approved local business transportation expenses must be submitted within "
             "15 calendar days after the expense date."),
            ("5. Documentation",
             "All reimbursement claims must include original receipts or valid digital invoices."),
        ],
    },
}


def create_pdf(filename, document_data):
    output_path = OUTPUT_DIR / filename

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=55,
        leftMargin=55,
        topMargin=55,
        bottomMargin=55,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocumentTitle",
        parent=styles["Title"],
        fontSize=17,
        leading=22,
        alignment=TA_CENTER,
        textColor=colors.darkblue,
        spaceAfter=20,
    )

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=12,
        leading=16,
        spaceBefore=12,
        spaceAfter=6,
        textColor=colors.darkblue,
    )

    body_style = ParagraphStyle(
        "PolicyBody",
        parent=styles["BodyText"],
        fontSize=10,
        leading=15,
        spaceAfter=8,
    )

    story = [
        Paragraph(document_data["title"], title_style),
        Spacer(1, 10),
        Paragraph(
            "Document type: Internal Company Policy | "
            "Company: Apex Technologies | "
            "Version: 1.0",
            body_style,
        ),
    ]

    for heading, content in document_data["sections"]:
        story.append(Paragraph(heading, heading_style))
        story.append(Paragraph(content, body_style))

    doc.build(story)
    print(f"Created: {output_path}")


def main():
    for filename, document_data in DOCUMENTS.items():
        create_pdf(filename, document_data)

    print("\nBenchmark v2 policy documents created successfully.")
    print(f"Total documents: {len(DOCUMENTS)}")


if __name__ == "__main__":
    main()
