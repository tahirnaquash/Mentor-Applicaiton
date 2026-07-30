import models
from database import SessionLocal

db = SessionLocal()
try:
    # 1. Check if Global configuration exists
    global_config = db.query(models.SystemConfiguration).filter(models.SystemConfiguration.department == "Global").first()
    if not global_config:
        print("Seeding Global configuration thresholds...")
        db.add(models.SystemConfiguration(
            department="Global",
            attendance_threshold=75.0,
            cgpa_threshold=6.5,
            term_name="Spring Semester 2026"
        ))

    # 2. Check if your specific Department configuration exists
    # Replace "Computer Science" with whatever your test admin's department is
    cs_config = db.query(models.SystemConfiguration).filter(models.SystemConfiguration.department == "Computer Science").first()
    if not cs_config:
        print("Seeding Computer Science configuration thresholds...")
        db.add(models.SystemConfiguration(
            department="Computer Science",
            attendance_threshold=75.0,
            cgpa_threshold=6.0,
            term_name="CS Term 2026"
        ))

    db.commit()
    print("✅ System configuration matrix seeded successfully!")
except Exception as e:
    db.rollback()
    print(f"❌ Error seeding configuration: {e}")
finally:
    db.close()