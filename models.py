import datetime
import enum
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, Float, DateTime, Enum, Text, UniqueConstraint
from sqlalchemy.orm import backref, relationship
from sqlalchemy.sql import func
from database import Base
from datetime import datetime, timezone

class DbUser(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    display_name = Column(String, nullable=True)
    department = Column(String, nullable=True)
    usn_or_id = Column(String(50), unique=True, index=True, nullable=True)
    mentor_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    assigned_students = relationship(
        "DbUser",
        backref="assigned_mentor",
        remote_side=[id]
    )

    student_profile = relationship(
        "DbStudentInfo",
        primaryjoin="DbUser.usn_or_id == foreign(DbStudentInfo.student_id)",
        back_populates="user_account",
        uselist=False,
        cascade="all, delete-orphan"
    )

    academic_records = relationship(
        "DbAcademicRecord",
        primaryjoin="DbUser.usn_or_id == foreign(DbAcademicRecord.student_id)",
        back_populates="user_account",
        cascade="all, delete-orphan"
    )

    course_enrollments = relationship(
        "DbCourseRegistration",
        primaryjoin="DbUser.usn_or_id == foreign(DbCourseRegistration.student_id)",
        backref="user"
    )

    academic_state = relationship(
        "StudentAcademicState",
        primaryjoin="DbUser.usn_or_id == foreign(StudentAcademicState.student_id)",
        foreign_keys="StudentAcademicState.student_id",
        uselist=False,
        back_populates="student_profile",
        cascade="all, delete-orphan"
    )

    term_mappings = relationship(
        "MentorStudentMapping",
        primaryjoin="DbUser.usn_or_id == foreign(MentorStudentMapping.student_usn_or_id)",
        foreign_keys="MentorStudentMapping.student_usn_or_id",
        back_populates="student",
        viewonly=True
    )


# =========================================================
#   REFACTORED STUDENT TABLES (ALIGNING FOREIGN KEY CORRIDORS)
# =========================================================

class DbStudentInfo(Base):
    __tablename__ = "student_info"

    # student_info.student_id should store the student's USN / usn_or_id
    student_id = Column(
        String(50),
        ForeignKey("users.usn_or_id", ondelete="CASCADE"),
        primary_key=True,
        unique=True,
        nullable=False
    )

    father_name = Column(String, nullable=True)
    mother_name = Column(String, nullable=True)
    guardian_name = Column(String, nullable=True)

    phone_number = Column(String, nullable=True)
    fathers_phone = Column(String, nullable=True)
    mothers_phone = Column(String, nullable=True)
    address = Column(String, nullable=True)

    tenth_percentage = Column(Float, nullable=True)
    twelfth_percentage = Column(Float, nullable=True)
    diploma_percentage = Column(Float, nullable=True)

    # Explicit join back to users.usn_or_id
    user_account = relationship(
        "DbUser",
        primaryjoin="foreign(DbStudentInfo.student_id) == DbUser.usn_or_id",
        back_populates="student_profile"
    )

class DbAcademicRecord(Base):
    __tablename__ = "academic_records"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String(50), ForeignKey("student_info.student_id", ondelete="CASCADE"), nullable=False)
    semester = Column(Integer, nullable=False)
    academic_year = Column(String(50), nullable=False)
    semester_gpa = Column(Float, default=0.0)
    cumulative_cgpa = Column(Float, default=0.0)
    overall_attendance = Column(Float, default=100.0)
    backlogs_count = Column(Integer, default=0)
    is_transferred_to_counselor = Column(Boolean, default=False, nullable=False)
    
    #user_account = relationship("DbUser", back_populates="student_profile")
    user_account = relationship(
        "DbUser", 
        primaryjoin="DbAcademicRecord.student_id == DbUser.usn_or_id",
        foreign_keys=[student_id],
        back_populates="academic_records"
    )
    # --- ADD THIS MISSING FLAG FIELD LINE BELOW ---
    is_historical_snapshot = Column(Boolean, default=False, nullable=False)

class EvaluationTypeEnum(str, enum.Enum):
    IA1 = "IA-1"
    IA2 = "IA-2"
    IA_AVG = "IA-AVG"
    SEE = "SEE"

class DbMentorDetail(Base):
    __tablename__ = "mentor_details"

    id = Column(Integer, primary_key=True, index=True)
    
    # Foreign Key establishing connection back to the 'users' table via usn_or_id
    employee_id = Column(String, ForeignKey("users.usn_or_id", ondelete="CASCADE"), unique=True, nullable=False)
    
    # Profile & Identity Fields
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    phone_number = Column(String(20), nullable=True)
    department_name = Column(String(100), nullable=False)
    designation = Column(String(100), nullable=False)
    gender = Column(String(20), nullable=True)
    highest_qualification = Column(String(100), nullable=True)
    
    # Financial & National Identifier Information (Handled as secure text strings)
    pan_card_number = Column(String(50), nullable=True)
    # [Aadhaar Redacted] - Field mapped structurally as a string placeholder for compliance
    aadhaar_card_number = Column(String(50), nullable=True) 
    
    # Tenure Timeline Fields
    date_of_joining = Column(DateTime, nullable=False)
    relieving_date = Column(DateTime, nullable=True)
    
    # Granular Work Experience Metrics (Stored as Decimals/Floats for precise year tracking)
    experience_in_college = Column(Float, default=0.0, nullable=False)
    teaching_experience_years = Column(Float, default=0.0, nullable=False)
    research_experience_years = Column(Float, default=0.0, nullable=False)
    industry_experience_years = Column(Float, default=0.0, nullable=False)
    other_experience_years = Column(Float, default=0.0, nullable=False)
    total_work_experience_years = Column(Float, default=0.0, nullable=False)

    # Bi-directional relationship back to the master DbUser table
    user_account = relationship(
        "DbUser", 
        backref=backref("mentor_profile", uselist=False, cascade="all, delete-orphan"),
        primaryjoin="DbMentorDetail.employee_id == DbUser.usn_or_id",
        foreign_keys=[employee_id]
    )

class DbCourse(Base):
    """Global inventory catalog of courses across all semesters."""
    __tablename__ = "courses"
    
    id = Column(Integer, primary_key=True, index=True)
    course_code = Column(String, unique=True, index=True, nullable=False)
    course_name = Column(String, nullable=False)
    assigned_semester = Column(Integer, nullable=False) # 1 to 8
    department = Column(String, nullable=False)


class DbCourseRegistration(Base):
    __tablename__ = "course_registrations"

    id = Column(Integer, primary_key=True, index=True)
    
    # Matches: FOREIGN KEY (student_id) REFERENCES student_info(student_id) ON DELETE CASCADE
    student_id = Column(String(50), ForeignKey("student_info.student_id", ondelete="CASCADE"), nullable=False)
    
    # Matches: FOREIGN KEY (course_id) REFERENCES courses(course_code) ON DELETE CASCADE
    course_id = Column(String(50), ForeignKey("courses.course_code", ondelete="CASCADE"), nullable=False)
    
    # Matches: semester INTEGER NOT NULL
    semester = Column(Integer, nullable=False)
    
    # Matches: academic_year VARCHAR(50) NOT NULL
    academic_year = Column(String(50), nullable=False)

    # Table Argument block to enforce structural integrity constraints at the ORM layer
    __table_args__ = (
        # Matches: CONSTRAINT unique_student_course_term UNIQUE (student_id, course_id, academic_year, semester)
        UniqueConstraint(
            'student_id', 'course_id', 'academic_year', 'semester', 
            name='unique_student_course_term'
        ),
    )

    # =========================================================================
    #  RELATIONAL GRAPH LINKS
    # =========================================================================
    
    # # Keeps identity reference links running smoothly back to authentication node
    # student = relationship(
    #     "DbUser",
    #     primaryjoin="DbCourseRegistration.student_id == foreign(DbUser.usn_or_id)"
    # )
    
    # # Links cleanly to courses inventory using course_code as the tracking anchor
    # course = relationship(
    #     "DbCourse",
    #     primaryjoin="DbCourseRegistration.course_id == foreign(DbCourse.course_code)"
    # )
    #  Replace it with this:
    student = relationship(
        "DbUser",
        primaryjoin="DbCourseRegistration.student_id == foreign(DbUser.usn_or_id)",
        backref="registrations"
    )

    course = relationship(
        "DbCourse",
        primaryjoin="DbCourseRegistration.course_id == foreign(DbCourse.course_code)",
        uselist=False
    )
    # Downstream relationships cascade seamlessly on deletion events
    marks_ledger = relationship("DbMarksLedger", back_populates="registration", cascade="all, delete-orphan")

class DbMarksLedger(Base):
    """Stores target IA1, IA2, and SEE milestones alongside attendance tracking components."""
    __tablename__ = "marks_ledger"
    
    id = Column(Integer, primary_key=True, index=True)
    registration_id = Column(Integer, ForeignKey("course_registrations.id", ondelete="CASCADE"), nullable=False)
    
    evaluation_type = Column(
        Enum(
            EvaluationTypeEnum,
            name="evaluation_type_enum",
            values_callable=lambda enum_cls: [item.value for item in enum_cls],
        ),
        nullable=False
    )
    marks_obtained = Column(Float, nullable=True)
    max_marks = Column(Float, default=20.0)
    attendance_percentage = Column(Float, nullable=True)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    registration = relationship("DbCourseRegistration", back_populates="marks_ledger")

class StudentAcademicState(Base):
    __tablename__ = "student_academic_state"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String(50), ForeignKey("users.usn_or_id", ondelete="CASCADE"), unique=True, nullable=False)
    current_year = Column(Integer, default=1)
    current_semester = Column(Integer, default=1)
    last_advanced_academic_year = Column(String(50), default=None, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Use a string literal target pointing back to 'DbUser'
    student_profile = relationship("DbUser", back_populates="academic_state")

# ==========================================
#     EXISTING INFRASTRUCTURE TRACKING
# ==========================================

class SystemConfiguration(Base):
    __tablename__ = "system_configurations"
    id = Column(Integer, primary_key=True, index=True)
    department = Column(String, default="Global")
    attendance_threshold = Column(Float, default=75.0)
    cgpa_threshold = Column(Float, default=6.5)
    term_name = Column(String, default="Spring Semester 2026")

class PerformanceMetric(Base):
    __tablename__ = "performance_metrics"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, server_default=func.now())
    active_connections = Column(Integer)
    response_time_ms = Column(Float)

class CounselingRecord(Base):
    __tablename__ = "counseling_records"
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String(50), ForeignKey("users.usn_or_id", ondelete="CASCADE"))
    counselor_id = Column(String(50), ForeignKey("users.usn_or_id", ondelete="CASCADE"))
    created_at = Column(DateTime, server_default=func.now())
    encrypted_session_notes = Column(String)
    summary_flag = Column(String)

# Inside models.py
class AcademicTermControl(Base):
    __tablename__ = "academic_term_controls" # Or your exact database table name

    id = Column(Integer, primary_key=True, index=True)
    department = Column(String(100), nullable=False)
    semester = Column(String(50), nullable=False)
    academic_year = Column(String(50), nullable=False)

class MentorStudentMapping(Base):
    __tablename__ = "mentor_student_mappings"

    id = Column(Integer, primary_key=True, index=True)
    
    # 1. FIXED: Change Integer to String(255) to support your actual VARCHAR database schema
    mentor_id = Column(String(255), nullable=False) 
    student_usn_or_id = Column(String(50), ForeignKey("users.usn_or_id", ondelete="CASCADE"), nullable=False)
    academic_year = Column(String(50), nullable=False)
    semester_type = Column(String(20), nullable=False)
    is_active = Column(Boolean, default=True)
    assigned_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        UniqueConstraint(
            "student_usn_or_id", "academic_year", "semester_type",
            name="unique_term_mapping"
        ),
    )

    # 2. FIXED: Explicitly specify a primaryjoin expression since mentor_id is a VARCHAR string pointing to usn_or_id
    mentor_user = relationship(
        "DbUser", 
        primaryjoin="MentorStudentMapping.mentor_id == DbUser.usn_or_id",
        foreign_keys=[mentor_id],
        viewonly=True
    )
    
    # Keep your student relationship intact
    student = relationship("DbUser", foreign_keys=[student_usn_or_id], back_populates="term_mappings")

# Inside models.py
# Leave only ONE of these blocks inside models.py
class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    operator_email = Column(String(255), nullable=False)
    event_type = Column(String(100), nullable=False)
    severity = Column(String(20), nullable=False)
    affected_count = Column(Integer, default=0)
    details = Column(Text, nullable=False)
    ip_address = Column(String(45), nullable=False)
    executed_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

from sqlalchemy.dialects.postgresql import ARRAY

# Add this class at the bottom of models.py
from sqlalchemy import Column, Integer, String, Text
from database import Base

class DbClinicalKnowledgeBase(Base):
    __tablename__ = "clinical_knowledge_base"

    id = Column(Integer, primary_key=True, index=True)
    domain = Column(String(50), nullable=False)
    keyword_combinations = Column(Text, nullable=False)
    predefined_title = Column(String(255), nullable=False)
    predefined_answer = Column(Text, nullable=False)
    severity_tier = Column(Integer, default=1)
    action_url = Column(Text, nullable=True)

from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, func
from database import Base # Assuming your base setup looks like this


# ==========================================
# SQLALCHEMY DATABASE ORM LAYER
# ==========================================
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
# ... other imports ...

class MentoringReport(Base):
    __tablename__ = "mentoring_reports"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String, ForeignKey("users.usn_or_id"), nullable=False)
    mentor_id = Column(String, ForeignKey("users.usn_or_id"), nullable=False)
    discussion_points = Column(Text, nullable=False)
    action_items = Column(Text, nullable=True)
    meeting_date = Column(DateTime, default=datetime.utcnow)
    
    # ─── ADD THESE TWO COLUMNS HERE ──────────────────────────────────
    current_semester = Column(Integer, nullable=False, default=1)
    academic_year = Column(String, nullable=False) 
    # ─────────────────────────────────────────────────────────────────

    # Establish relationship back to the main user accounts table matrix
    student = relationship(
        "DbUser", 
        primaryjoin="DbUser.usn_or_id == MentoringReport.student_id"
    )

# ==========================================
# PYDANTIC INCOMING DATA VALIDATORS
# ==========================================
class MentoringReportCreate(BaseModel):
    student_id: str = Field(..., max_length=50)
    mentor_id: str = Field(..., max_length=50)
    academic_year: str = Field(..., max_length=20)
    current_semester: int = Field(..., ge=1, le=8)
    discussion_points: str
    action_items: str | None = None

    