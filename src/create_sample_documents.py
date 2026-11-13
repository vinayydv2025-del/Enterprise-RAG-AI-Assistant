
from pathlib import Path
from html import escape
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

styles = getSampleStyleSheet()

documents = {
    "Employee_HR_Policy.pdf": {
        "title": "Employee HR Policy",
        "sections": [
            ("Company", "Northstar Solutions is a fictional technology company used for this AI project."),
            ("Working Hours", "Standard working hours are 9:00 AM to 6:00 PM, Monday through Friday."),
            ("Remote Work", "Employees may work remotely up to two days per week with manager approval."),
            ("Probation", "New employees have a probation period of six months."),
            ("Performance Reviews", "Employee performance reviews take place every six months."),
            ("Employee Benefits", "Eligible full-time employees receive health insurance and professional development support.")
        ]
    },

    "Information_Security_Policy.pdf": {
        "title": "Information Security Policy",
        "sections": [
            ("Password Requirements", "Passwords must contain at least 12 characters, including uppercase letters, lowercase letters, numbers and special characters."),
            ("Multi-Factor Authentication", "Multi-factor authentication is mandatory for company email, cloud platforms and internal business applications."),
            ("Data Access", "Employees may access only the information required for their assigned job responsibilities."),
            ("Data Sharing", "Confidential company information must not be shared with unauthorized individuals or personal email accounts."),
            ("Security Incidents", "Suspected security incidents must be reported to the IT security team within one hour of discovery."),
            ("Device Security", "Company laptops must use full-disk encryption and automatic screen locking.")
        ]
    },

    "Leave_and_Attendance_Policy.pdf": {
        "title": "Leave and Attendance Policy",
        "sections": [
            ("Annual Leave", "Full-time employees receive 20 days of paid annual leave per calendar year."),
            ("Sick Leave", "Employees receive 10 days of paid sick leave per calendar year."),
            ("Public Holidays", "The company observes 12 public holidays each calendar year."),
            ("Leave Approval", "Planned leave requests should be submitted at least five working days in advance."),
            ("Emergency Leave", "Employees should notify their manager as soon as reasonably possible when emergency leave is required."),
            ("Attendance", "Employees are expected to record their attendance through the company's attendance system."),
            ("Late Arrival", "Employees arriving more than 15 minutes late should notify their manager.")
        ]
    }
}


def create_pdf(filename, title, sections):
    file_path = OUTPUT_DIR / filename

    story = [
        Paragraph(escape(title), styles["Title"]),
        Spacer(1, 20),
        Paragraph(
            "Northstar Solutions - Internal Knowledge Document",
            styles["Normal"]
        ),
        Spacer(1, 15)
    ]

    for heading, content in sections:
        story.append(
            Paragraph(escape(heading), styles["Heading2"])
        )
        story.append(
            Paragraph(escape(content), styles["BodyText"])
        )
        story.append(Spacer(1, 12))

    pdf = SimpleDocTemplate(
        str(file_path),
        pagesize=A4
    )

    pdf.build(story)
    print(f"Created: {file_path.name}")


if __name__ == "__main__":
    for filename, document in documents.items():
        create_pdf(
            filename,
            document["title"],
            document["sections"]
        )

    print("\nAll sample PDF documents created successfully!")
