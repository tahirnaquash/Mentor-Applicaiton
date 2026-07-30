from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class UserLoginSchema(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=100)

class UserCreateSchema(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=100)
    role: str = Field(..., description="Must be admin, mentor, student, or counselor")
    display_name: str = Field(..., min_length=2, max_length=100)
    department: Optional[str] = None
    usn_or_id: Optional[str] = None

class MappingRequestSchema(BaseModel):
    student_id: int
    mentor_id: int

class ConfigUpdateSchema(BaseModel):
    attendance_threshold: float = Field(..., ge=0.0, le=100.0)
    cgpa_threshold: float = Field(..., ge=0.0, le=10.0)
    term_name: str = Field(..., min_length=2, max_length=50)

class ThresholdConfigSchema(BaseModel):
    attendance_threshold: float = Field(..., ge=0.0, le=100.0)
    cgpa_threshold: float = Field(..., ge=0.0, le=10.0)
    term_name: str = Field(..., min_length=2, max_length=100)

class ReallocateCohortSchema(BaseModel):
    source_mentor_id: int
    target_mentor_id: int