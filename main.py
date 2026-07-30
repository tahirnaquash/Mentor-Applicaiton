from urllib import request

from fastapi import FastAPI, Request, Form, Depends, HTTPException, UploadFile, File, status
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from starlette.exceptions import HTTPException as StarletteHTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address
from datetime import datetime
from urllib.parse import urlencode
from dependencies import get_current_user
from security import hash_password, verify_session_token
from sqlalchemy.orm import joinedload
from security import create_session_token
from typing import List, Optional
import csv
import io

import models
import schemas
from database import Base, engine, get_db
from security import verify_password, hash_password
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer
from database import SessionLocal, get_db
import models
import numpy as np
from slowapi import Limiter
from slowapi.util import get_remote_address

# 1. Define the Lifespan Context Manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup configuration and memory warming, 
    then cleans up automatically when the server shuts down.
    """
    print("Application booting up... warming up semantic matrix memory arrays.")
    
    # Open a direct database session for initialization
    db = SessionLocal()
    try:
        # Pre-warm our global memory cache before accepting traffic
        ensure_semantic_cache(db)
    except Exception as startup_err:
        print(f"[CRITICAL STARTUP ERROR] Cache pre-warm failed: {startup_err}")
    finally:
        db.close()
        
    print("Production server is fully warmed up and listening for requests.")
    
    # Everything BEFORE 'yield' runs on Startup
    yield
    # Everything AFTER 'yield' runs on Shutdown
    
    print("Shutting down application... cleaning up global memory state.")
    global DB_RECORDS_CACHE, CACHED_EMBEDDINGS
    DB_RECORDS_CACHE.clear()
    CACHED_EMBEDDINGS = None

# 2. Pass the lifespan handler directly into your FastAPI app instance
app = FastAPI(lifespan=lifespan)

# =========================================================================
# GLOBAL CACHE & LOGIC MATCHING ENGINE (Remains unchanged)
# =========================================================================
ml_model = None
DB_RECORDS_CACHE = []
CACHED_EMBEDDINGS = None

def ensure_semantic_cache(db: Session):
    """Loads all rows and bakes their ML embeddings into memory once."""
    global DB_RECORDS_CACHE, CACHED_EMBEDDINGS, ml_model
    
    if ml_model is None:
        try:
            ml_model = SentenceTransformer('local_model_files', model_kwargs={"local_files_only": True})
        except Exception:
            ml_model = SentenceTransformer('all-MiniLM-L6-v2')
            ml_model.save('local_model_files')

    if not DB_RECORDS_CACHE:
        print("Baking 3,000 matrix rows into semantic memory space...")
        DB_RECORDS_CACHE = db.query(models.DbClinicalKnowledgeBase).all()
        if DB_RECORDS_CACHE:
            corpus = [f"{r.predefined_title} {r.keyword_combinations}" for r in DB_RECORDS_CACHE]
            CACHED_EMBEDDINGS = ml_model.encode(corpus, convert_to_numpy=True)
            print(f"Successfully cached {len(DB_RECORDS_CACHE)} database rows in memory.")

# =========================================================================
# 2. DEFINE THE LIFESPAN FUNCTION NEXT
# =========================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Booting up backend server layers...")
    db = SessionLocal()
    try:
        ensure_semantic_cache(db)
    except Exception as e:
        print(f"[STARTUP EXCEPTION] Warmup sequence paused: {e}")
    finally:
        db.close()
    yield
    print("Cleaning up system memory allocations.")

# =========================================================================
# 3. NOW INITIALIZE THE APP AND THE LIMITER
# =========================================================================
limiter = Limiter(key_func=get_remote_address)

# Python can now safely read 'lifespan' because it is declared right above!
app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
templates = Jinja2Templates(directory="templates")

# =========================================================================
#  DEPENDENCY INJECTIONS & UTILITIES
# =========================================================================
async def get_current_active_admin(request: Request, db: Session = Depends(get_db)):
    """Extracts current user session from request tracking token cookie context."""
    token = request.cookies.get("session_token")
    if not token:
        raise HTTPException(status_code=401, detail="Unauthorized session context.")
    user = db.query(models.DbUser).filter(models.DbUser.email == token).first()
    if not user or user.role not in ["admin", "superadmin"]:
        raise HTTPException(status_code=403, detail="Access Denied.")
    return user


# async def get_current_active_admin(request: Request, db: Session = Depends(get_db)):
#     """Extracts current user session from request tracking token cookie context."""
#     token = request.cookies.get("session_token")
#     if not token:
#         raise HTTPException(status_code=401, detail="Unauthorized session context.")
        
#     try:
#         # Decrypt/verify the cryptographic token to extract the true user key
#         decrypted_user_id = verify_session_token(token)
#         # Query using the decoded integer ID field rather than plain email address fields
#         user = db.query(models.DbUser).filter(models.DbUser.email == token).first()

#     except Exception:
#         user = None

#     if not user or user.role not in ["admin", "superadmin"]:
#         raise HTTPException(status_code=403, detail="Access Denied.")
        
#     return user

def get_active_mapping_term(db: Session, current_admin: models.DbUser):
    """Resolve the active mentor mapping term, falling back to the app default term."""
    target_dept = current_admin.department if current_admin.role != "superadmin" else "Global Science"
    term = db.query(models.AcademicTermControl).filter(
        models.AcademicTermControl.department == target_dept
    ).order_by(models.AcademicTermControl.id.desc()).first()

    if not term and current_admin.role == "superadmin":
        term = db.query(models.AcademicTermControl).order_by(models.AcademicTermControl.id.desc()).first()

    if term:
        return term.academic_year, term.semester
    return "2026-2027", "Odd"


# =========================================================================
#        PUBLIC CORRIDORS & LANDING
# =========================================================================

@app.get("/", response_class=HTMLResponse)
async def platform_landing_index(request: Request):
    """Renders the public core entry portal."""
    mock_marketing_data = {
        "showcase_mentors": [
            {"name": "Sarah Jenkins", "dept": "Senior Web Developer"},
            {"name": "Dr. Alex Rivera", "dept": "AI Research Scientist"},
            {"name": "James Chen", "dept": "UI/UX Design Lead"},
            {"name": "Elena Rostova", "dept": "Machine Learning Engineer"}
        ]
    }
    return templates.TemplateResponse(
        request=request,
        name="landing.html",
        context={"request": request, "data": mock_marketing_data}
    )


# =========================================================================
#          AUTHENTICATION VAULTS & ROUTERS
# =========================================================================

@app.get("/auth/login/admin", response_class=HTMLResponse)
async def admin_login_view(request: Request):
    return templates.TemplateResponse(request=request, name="login_admin.html", context={"request": request})

@app.get("/auth/login/mentor", response_class=HTMLResponse)
async def mentor_login_view(request: Request):
    return templates.TemplateResponse(request=request, name="login_mentor.html", context={"request": request})

@app.get("/auth/login/student", response_class=HTMLResponse)
async def student_login_view(request: Request):
    return templates.TemplateResponse(request=request, name="login_student.html", context={"request": request})

@app.get("/auth/login/counselor", response_class=HTMLResponse)
async def counselor_login_view(request: Request):
    return templates.TemplateResponse(request=request, name="login_counselor.html", context={"request": request})


@app.post("/auth/login/admin")
async def handle_admin_login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    """Queries DbUser, verifies credentials, and securely drops a mock tracking session token."""
    user = db.query(models.DbUser).filter(models.DbUser.email == username).first()
    
    if not user or user.role not in ["admin", "superadmin"] or not user.is_active or not verify_password(password, user.hashed_password):
        return templates.TemplateResponse(
            request=request,
            name="login_admin.html",
            context={"request": request, "error": "Username / password is incorrect or contact administrator."}
        )

    response = RedirectResponse(url="/dashboard/admin?msg=Root+session+authenticated+successfully.", status_code=303)
    response.set_cookie(key="session_token", value=username, httponly=True, max_age=3600)
    return response




@app.post("/auth/login/mentor")
async def handle_mentor_login(
    request: Request,
    username: str = Form(...),  # Matches name="username" field in your HTML
    password: str = Form(...),  # Matches name="password" field in your HTML
    db: Session = Depends(get_db)
):
    """
    Validates faculty logins securely and issues localized viewport session scopes.
    """
    # 1. Look up user record verifying role constraint rules
    user = db.query(models.DbUser).filter(
        models.DbUser.email == username.strip().lower(),
        models.DbUser.role == "mentor"
    ).first()

    # 2. Check credentials
    if not user or not verify_password(password, user.hashed_password):
        return RedirectResponse(
            url="/auth/login/mentor?err=Invalid+Credentials+or+Unauthorized",
            status_code=303
        )

    # 3. Create persistent security session cookie state pointing to the user email or identifier
    response = RedirectResponse(url="/dashboard/mentor", status_code=303)
    response.set_cookie(
        key="session_token",
        value=user.email, # Storing email matching your get dashboard lookup criteria
        httponly=True,
        max_age=28800, # 8 Hours session persistence
        samesite="lax"
    )
    return response


@app.post("/auth/login/student")
async def handle_student_login(
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    # Query user by email (case‑insensitive)
    user = db.query(models.DbUser).filter(
        models.DbUser.email == username
    ).first()
    # Ensure user exists, is active, and has student role (case‑insensitive)
    if not user or not user.is_active or user.role.lower() != "student":
        return RedirectResponse(
            url="/auth/login/student?err=Invalid+Credentials+or+Unauthorized",
            status_code=303
        )
    # Verify password
    if not verify_password(password, user.hashed_password):
        return RedirectResponse(
            url="/auth/login/student?err=Invalid+Credentials+or+Unauthorized",
            status_code=303
        )
    # On success, issue session cookie
    response = RedirectResponse(url="/dashboard/student", status_code=303)
    response.set_cookie(
        key="session_token",
        value=user.email,
        httponly=True,
        max_age=28800,
        samesite="lax"
    )
    return response



@app.post("/auth/login/counselor")
async def handle_counselor_login(request: Request, username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(models.DbUser).filter(models.DbUser.email == username).first()
    if not user or user.role != "counselor" or not user.is_active or not verify_password(password, user.hashed_password):
        return templates.TemplateResponse(request=request, name="login_counselor.html", context={"request": request, "error": "Invalid Credentials"})
    return RedirectResponse(url="/dashboard/counselor", status_code=303)


@app.post("/auth/dashboard/change-password")
async def account_self_password_mutation(
    email_key: str = Form(...),
    current_password: str = Form(...),
    new_password: str = Form(...),
    db: Session = Depends(get_db)
):
    """Self-service credential mutation tracking context."""
    user = db.query(models.DbUser).filter(models.DbUser.email == email_key).first()
    if not user or not verify_password(current_password, user.hashed_password):
        return RedirectResponse(
            url=f"/dashboard/{user.role if user else 'admin'}?err=Authentication+Failure:+Current+password+is+incorrect.", 
            status_code=303
        )
        
    try:
        user.hashed_password = hash_password(new_password)
        db.commit()
        return RedirectResponse(url=f"/dashboard/{user.role}?msg=Credentials+successfully+updated.", status_code=303)
    except Exception as e:
        db.rollback()
        return RedirectResponse(url=f"/dashboard/{user.role}?err=Mutation+Fault:+{str(e)}", status_code=303)


@app.get("/auth/logout")
async def terminate_user_session():
    response = RedirectResponse(url="/", status_code=303)
    response.delete_cookie(key="session_token")
    response.delete_cookie(key="access_token")
    return response


# =========================================================================
#   MANUAL IDENTITY ACCOUNT CREATION & MANAGEMENT
# =========================================================================

@app.post("/admin/users/create")
@app.post("/admin/config/create-user")
async def provision_account_node(
    display_name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form(...),
    department: str = Form(None),
    usn_or_id: str = Form(None),
    db: Session = Depends(get_db),
    request: Request = None
):
    """
    Provisioning Accounts: Securely hashes passwords, inserts new user rows natively, 
    and handles automatic lifecycle instantiation for student entities.
    """
    user_token = request.cookies.get("session_token")
    if not user_token:
        return RedirectResponse(url="/auth/login/admin?err=Session+Expired", status_code=303)
        
    # --- FIXED SESSION RESOLUTION ---
    # Decodes and verifies token signature natively rather than matching email against raw payload
    decrypted_user_id = verify_session_token(user_token)
    if not decrypted_user_id:
        return RedirectResponse(url="/auth/login/admin?err=Invalid+Session+Token", status_code=303)
        
    current_admin = db.query(models.DbUser).filter(models.DbUser.id == decrypted_user_id).first()
    if not current_admin or current_admin.role not in ["admin", "superadmin"]:
        return RedirectResponse(url="/auth/login/admin?err=Unauthorized", status_code=303)

    try:
        # Prevent Duplicate Email Signups
        existing_user = db.query(models.DbUser).filter(models.DbUser.email == email).first()
        if existing_user:
            return RedirectResponse(url="/dashboard/admin?action=create_user&err=Account+already+exists.", status_code=303)

        # Enforce Multi-Tenant Department Boundaries
        target_dept = department
        if current_admin.role != "superadmin":
            target_dept = current_admin.department
            if role in ["superadmin", "admin", "counselor"]:
                return RedirectResponse(url="/dashboard/admin?action=create_user&err=Security+Violation:+Insufficient+clearance.", status_code=303)

        # Safeguard: Require a Registration ID / USN if provisioning a student profile
        if role == "student" and not usn_or_id:
            return RedirectResponse(url="/dashboard/admin?action=create_user&err=Validation+Error:+Students+require+a+valid+USN/Reg+ID.", status_code=303)

        # 1. Store Base User Entity
        hashed_pass = hash_password(password)
        new_user = models.DbUser(
            display_name=display_name,
            email=email,
            hashed_password=hashed_pass,
            role=role,
            department=target_dept,
            usn_or_id=usn_or_id,
            is_active=True
        )
        db.add(new_user)
        db.flush() # Flush extracts the primary key data structures without closing the atomic transaction early

        # 2. CONDITIONAL ACADEMIC TRACKING INJECTION FOR STUDENTS
        if role == "student":
            # Direct SQL execution provides absolute layout consistency and ensures parameters like DEFAULT metrics match the DDL blueprint exactly
            db.execute(
                text("""
                    INSERT INTO student_academic_state (student_id, current_year, current_semester, last_advanced_academic_year, updated_at)
                    VALUES (:student_id, 1, 1, NULL, NOW());
                """),
                {"student_id": usn_or_id}
            )

        # Commit both operations safely in a single transaction block
        db.commit()
        return RedirectResponse(url="/dashboard/admin?action=create_user&tab=provision&msg=Identity+and+academic+ledger+records+successfully+committed.", status_code=303)
        
    except Exception as e:
        db.rollback()  
        return RedirectResponse(url=f"/dashboard/admin?err=Database+Insertion+Fault:+{str(e)}", status_code=303)

@app.post("/admin/users/update-password")
@app.post("/admin/config/change-password")
async def administrative_password_override_endpoint(
    request: Request,
    target_email: str = Form(None, alias="target_email"),
    identity_key: str = Form(None, alias="identity_key"),
    new_password: str = Form(...),
    db: Session = Depends(get_db)
):
    """Secure password reset enforcing multi-tenant role and department limits."""
    user_token = request.cookies.get("session_token")
    if not user_token:
        return RedirectResponse(url="/auth/login/admin?err=Session+Expired", status_code=303)
        
    current_admin = db.query(models.DbUser).filter(models.DbUser.email == user_token).first()
    if not current_admin or current_admin.role not in ["admin", "superadmin"]:
        return RedirectResponse(url="/auth/login/admin?err=Unauthorized", status_code=303)

    try:
        lookup_key = (target_email or identity_key or "").strip()
        target_user = db.query(models.DbUser).filter(
            (models.DbUser.email == lookup_key) | (models.DbUser.usn_or_id == lookup_key)
        ).first()
        
        if not target_user:
            return RedirectResponse(url="/dashboard/admin?action=change_password&err=Target+user+not+found", status_code=303)

        if current_admin.role != "superadmin":
            if target_user.department != current_admin.department:
                return RedirectResponse(
                    url="/dashboard/admin?action=change_password&err=Security+Violation:+User+is+outside+your+department.", 
                    status_code=303
                )
            if target_user.role not in ["student", "mentor"]:
                return RedirectResponse(
                    url="/dashboard/admin?action=change_password&err=Security+Violation:+You+can+only+modify+students+or+mentors.", 
                    status_code=303
                )

        target_user.hashed_password = hash_password(new_password.strip())
        db.commit()

        return RedirectResponse(url="/dashboard/admin?action=change_password&msg=Password+successfully+updated.", status_code=303)
        
    except Exception as e:
        db.rollback()
        return RedirectResponse(url=f"/dashboard/admin?err=Credential+Update+Fault:+{str(e)}", status_code=303)


@app.post("/admin/users/toggle-active/{user_id}")
async def enforce_role_session_status(user_id: int, db: Session = Depends(get_db), request: Request = None):
    """Enforces active profile connectivity state restrictions natively."""
    user_token = request.cookies.get("session_token")
    if not user_token:
        return RedirectResponse(url="/auth/login/admin?err=Session+Expired", status_code=303)
        
    current_admin = db.query(models.DbUser).filter(models.DbUser.email == user_token).first()
    target_user = db.query(models.DbUser).filter(models.DbUser.id == user_id).first()
    
    if not target_user:
         return RedirectResponse(url="/dashboard/admin?err=User+not+found", status_code=303)
         
    if current_admin.role != "superadmin" and target_user.department != current_admin.department:
        return RedirectResponse(url="/dashboard/admin?err=Access+Denied:+Target+is+in+another+department.", status_code=303)
        
    target_user.is_active = not target_user.is_active
    db.commit()
    
    return RedirectResponse(url=f"/dashboard/admin?tab=role_enforcement&msg=User+status+updated+to+{target_user.is_active}.", status_code=303)

# @app.get("/admin/config")
# async def get_admin_config_dashboard(
#     request: Request,
#     action: str = None,
#     status_filter: str = "unassigned",
#     dept_filter: str = None,
#     sem_filter: str = None,
#     ay_filter: str = "2026-2027",
#     db: Session = Depends(get_db),
#     current_admin: models.DbUser = Depends(get_current_active_admin)
# ):
#     # current_admin is guaranteed to be admin or superadmin
#     # 1. ENFORCE ROLE-BASED DEPARTMENT PRIVILEGES
#     if current_admin.role == "superadmin":
#         # Superadmins can filter by any department; if none specified, fetch all
#         target_department = dept_filter if dept_filter and dept_filter != "all" else None
#     else:
#         # Department Admins are locked to their own department (No bypass possible)
#         target_department = current_admin.department

#     # 2. CONSTRUCT DYNAMIC QUERY FOR STUDENTS
#     student_query = db.query(models.DbUser).filter(models.DbUser.role == "student")

#     if target_department:
#         student_query = student_query.filter(models.DbUser.department == target_department)
        
#     # Apply dynamic filter constraints
#     if status_filter == "assigned":
#         student_query = student_query.filter(models.DbUser.mentor_id.isnot(None))
#     elif status_filter == "unassigned":
#         student_query = student_query.filter(models.DbUser.mentor_id.is_(None))

#     # Optional: If you track current student semester/academic year on the users table
#     # (Otherwise, this filters down the student base pool dynamically)
#     # if sem_filter and sem_filter != 'all':
#     #     student_query = student_query.filter(models.DbUser.current_semester == int(sem_filter))

#     filtered_students = student_query.all()

#     # 3. CONSTRUCT QUERY FOR FACULTY/MENTORS
#     faculty_query = db.query(models.DbUser).filter(models.DbUser.role == "faculty") # or 'mentor'
#     if current_admin.role != "superadmin":
#         faculty_query = faculty_query.filter(models.DbUser.department == current_admin.department)
    
#     faculty_list = faculty_query.all()

#     # Fetch unique departments list for superadmin filter menu options
#     all_departments = []
#     if current_admin.role == "superadmin":
#         all_departments = [r[0] for r in db.query(models.DbUser.department).filter(models.DbUser.department.isnot(None)).distinct().all()]

#     filters = {
#         "status_filter": status_filter,
#         "dept_filter": dept_filter,
#         "sem_filter": sem_filter,
#         "ay_filter": ay_filter
#     }

#     return templates.TemplateResponse(
#     name="admin.html", 
#     context={
#         "request": request,
#         "student_list": filtered_students,
#         "faculty_list": faculty_list,
#         "all_departments": all_departments,
#         "current_user": current_admin,
#         "filters": filters
#     }
# )
# =========================================================================
#   BULK DATA CONTEXT MATRIX OPERATIONS (CSV IMPORTS)
# =========================================================================
@app.post("/admin/config/upload-csv")
@limiter.limit("10/minute")
async def bulk_upload_system_profiles_matrix(
    request: Request,
    upload_type: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: models.DbUser = Depends(get_current_active_admin)
):
    if not file:
        return JSONResponse(
            status_code=400, 
            content={"success": False, "msg": "No database CSV matrix uploaded."}
        )

    # SAFE IDENTITY EXTRACTION
    operator_email = None
    if current_user and hasattr(current_user, 'email') and current_user.email:
        operator_email = current_user.email
    else:
        operator_email = request.cookies.get("session_token")
    
    if not operator_email:
        operator_email = "system_admin@university.internal"

    try:
        contents = await file.read()
        buffer = io.StringIO(contents.decode('utf-8'))
        reader = csv.DictReader(buffer)
        
        total_updated_records = 0
        boundary_violations = []
        
        for row in reader:
            clean_row = {k.strip(): v.strip() for k, v in row.items() if k}
            row_dept = clean_row.get("department", "")
            display_name = clean_row.get("display_name", "Unknown Node")
            email_target = clean_row.get("email", "")
            usn_or_id = clean_row.get("usn_or_id", "")

            # MULTI-TENANT BOUNDARY CHECK
            if current_user and current_user.role != "superadmin":
                if not row_dept or row_dept != current_user.department:
                    boundary_violations.append({
                        "name": display_name,
                        "email": email_target if email_target else "N/A",
                        "target_dept": row_dept or "Not Specified"
                    })
                    continue 
            
            if not email_target:
                continue

            # FIX 1: NORMALIZE ROLE MATCHING (Handles both legacy HTML value & JS template strings)
            csv_explicit_role = clean_row.get("role", "").strip().lower()
            if csv_explicit_role in ["student", "mentor", "counselor", "admin", "superadmin"]:
                assigned_role = csv_explicit_role
            else:
                assigned_role = "student" if upload_type in ["student", "student_data"] else "mentor"

            # Enforce constraints: Student entities must provide a valid primary tracking identifier
            if assigned_role == "student" and not usn_or_id:
                boundary_violations.append({
                    "name": display_name,
                    "email": email_target,
                    "target_dept": "MISSING_REQUIRED_USN"
                })
                continue

            existing_user = db.query(models.DbUser).filter(models.DbUser.email == email_target).first()
            
            if existing_user:
                existing_user.display_name = clean_row.get("display_name", existing_user.display_name)
                existing_user.usn_or_id = usn_or_id if usn_or_id else existing_user.usn_or_id
                existing_user.department = row_dept if row_dept else existing_user.department
                # FIX 2: Ensure existing record matches the targeted batch operation type
                existing_user.role = assigned_role
            else:
                fallback_pass = hash_password(clean_row.get("password", "FallbackSecuredToken2026#"))
                new_user = models.DbUser(
                    email=email_target,
                    hashed_password=fallback_pass,
                    role=assigned_role,
                    display_name=clean_row.get("display_name"),
                    department=row_dept if row_dept else (current_user.department if current_user else "Computer Science"),
                    usn_or_id=usn_or_id,
                    is_active=True
                )
                db.add(new_user)
            
            # Flush changes to verify user primary/foreign key data constraints
            db.flush()

            # FIX 3: TARGETED STRUCTURAL UPDATE LOOP FOR ACADEMIC STATE
            if assigned_role == "student":
                try:
                    current_sem = int(clean_row.get("current_semester", 1))
                    current_yr = int(clean_row.get("current_year", 1))
                except (ValueError, TypeError):
                    current_sem = 1
                    current_yr = 1

                last_adv_year_raw = clean_row.get("last_advanced_academic_year", "")
                last_adv_year = int(last_adv_year_raw) if (last_adv_year_raw and last_adv_year_raw.isdigit()) else None

                # Perform a secure upsert verification routine
                from sqlalchemy import text
                db.execute(
                    text("""
                        INSERT INTO student_academic_state 
                            (student_id, current_year, current_semester, last_advanced_academic_year, updated_at)
                        VALUES 
                            (:student_id, :current_year, :current_semester, :last_advanced_academic_year, NOW())
                        ON CONFLICT (student_id) DO UPDATE 
                        SET 
                            current_year = EXCLUDED.current_year,
                            current_semester = EXCLUDED.current_semester,
                            last_advanced_academic_year = EXCLUDED.last_advanced_academic_year,
                            updated_at = NOW();
                    """),
                    {
                        "student_id": usn_or_id,
                        "current_year": current_yr,
                        "current_semester": current_sem,
                        "last_advanced_academic_year": last_adv_year
                    }
                )
                db.flush()
            
            total_updated_records += 1

        db.commit()
        
        # AUDIT LOGGING
        audit_entry = models.AuditLog(
            operator_email=operator_email,
            event_type="BATCH_CSV_IMPORT",
            severity="WARNING" if boundary_violations else "INFO",
            details=f"Asynchronous Processing complete. Saved: {total_updated_records}. Boundary Violations Dropped: {len(boundary_violations)}.",
            ip_address=request.client.host if request.client else "127.0.0.1"
        )
        db.add(audit_entry)
        db.commit()

        return JSONResponse(
            content={
                "success": True, 
                "msg": f"Sync routine evaluated successfully. Processed {total_updated_records} profiles.",
                "updated": total_updated_records,
                "violations_count": len(boundary_violations),
                "violations": boundary_violations
            }
        )
    
    except Exception as error_context:
        db.rollback()
        return JSONResponse(
            status_code=400,
            content={
                "success": False, 
                "msg": f"Matrix evaluation fault structural processing dropped: {str(error_context)}",
                "updated": 0,
                "violations_count": 0,
                "violations": []
            }
        )

@app.post("/admin/config/batch-import")
@limiter.limit("10/minute")
async def bulk_upload_system_profiles(
    request: Request,
    upload_type: str = Form(...),
    upload_subtype: str = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: models.DbUser = Depends(get_current_active_admin)
):
    """
    Ingests tabular matrix components into respective data catalogs safely.
    Natively enforces multi-tenant departmental separation constraints.
    """
    if not file or not file.filename.endswith('.csv'):
        return JSONResponse(
            status_code=400, 
            content={"success": False, "msg": "Dropped file format is invalid. Please drop a valid Tabular Matrix (.csv)."}
        )

    # Clean scope declarations to prevent UnboundLocalError instances
    total_updated_records = 0
    boundary_violations_skipped = 0

    try:
        contents = await file.read()
        # Decode utilizing 'utf-8-sig' to cleanly omit Excel BOM (Byte Order Mark) variations
        buffer = io.StringIO(contents.decode('utf-8-sig'))
        reader = csv.DictReader(buffer)
        
        if not reader.fieldnames:
            return JSONResponse(
                status_code=400, 
                content={"success": False, "msg": "Matrix ingestion aborted: CSV contains no valid header keys."}
            )

        # Iterate structural row data elements
        for row in reader:
            # Strip trailing white-spaces and convert keys to lower-case uniformly
            clean_row = {k.strip().lower(): (v.strip() if v else "") for k, v in row.items() if k}

            row_dept = clean_row.get("department", "")

            # Multi-Tenant Protection Boundary verification Check
            if upload_type in ["student_data", "mentor_data", "course_catalog"]:
                if current_user.role != "superadmin":
                    if row_dept and row_dept.lower() != current_user.department.lower():
                        boundary_violations_skipped += 1
                        continue

            # CASE 1: Student Directory Account Registration
            if upload_type == "student_data":
                email = clean_row.get("email", "")
                usn_or_id = clean_row.get("usn_or_id", "")
                if not email or not usn_or_id:
                    continue

                user_node = db.query(models.DbUser).filter(models.DbUser.email == email).first()
                if not user_node:
                    user_node = models.DbUser(
                        email=email,
                        hashed_password=hash_password(clean_row.get("password", "StudentPass2026#")),
                        role="student",
                        display_name=clean_row.get("display_name", "New Student Entry"),
                        department=row_dept if row_dept else current_user.department,
                        usn_or_id=usn_or_id,
                        is_active=True
                    )
                    db.add(user_node)
                    db.flush() # Ensure the tracking record exists for relations next

                # FIXED: Querying against 'student_id' instead of the non-existent 'usn_or_id' attribute
                student_info = db.query(models.DbStudentInfo).filter(models.DbStudentInfo.student_id == usn_or_id).first()
                if not student_info:
                    student_info = models.DbStudentInfo(student_id=usn_or_id)
                    db.add(student_info)

                student_info.father_name = clean_row.get("father_name", student_info.father_name)
                student_info.mother_name = clean_row.get("mother_name", student_info.mother_name)
                student_info.guardian_name = clean_row.get("guardian_name", student_info.guardian_name or "N/A")
                student_info.phone_number = clean_row.get("phone_number", student_info.phone_number)
                student_info.address = clean_row.get("address", student_info.address)
                
                # Defensive float parsing prevents casting crashes on empty values
                if clean_row.get("tenth_percentage"):
                    student_info.tenth_percentage = float(clean_row.get("tenth_percentage"))
                if clean_row.get("twelfth_percentage"):
                    student_info.twelfth_percentage = float(clean_row.get("twelfth_percentage"))
                if clean_row.get("diploma_percentage"):
                    student_info.diploma_percentage = float(clean_row.get("diploma_percentage"))
                
                total_updated_records += 1

            # CASE 2: Running Assessment Variables Update
            elif upload_type == "academic":
                usn_or_id = str(
                    clean_row.get("usn_or_id", clean_row.get("student_id", ""))
                ).strip().upper()
                print(f"Processing academic row for student ID: {usn_or_id}")
                if not usn_or_id:
                    continue

                # --------------------------------------------------
                # Student Validation
                # --------------------------------------------------
                student_user = db.query(models.DbUser).filter(
                    models.DbUser.usn_or_id == usn_or_id,
                    models.DbUser.role == "student"
                ).first()

                if not student_user:
                    continue

                if (
                    current_user.role != "superadmin"
                    and student_user.department.lower()
                    != current_user.department.lower()
                ):
                    boundary_violations_skipped += 1
                    continue

                # --------------------------------------------------
                # Semester / Academic Year
                # --------------------------------------------------
                semester_val = clean_row.get("semester")
                academic_year_val = str(
                    clean_row.get("academic_year", "")
                ).strip()

                try:
                    semester_int = int(float(str(semester_val).strip()))
                except Exception:
                    semester_int = None

                if semester_int is None:
                    continue

                # --------------------------------------------------
                # Evaluation Type
                # --------------------------------------------------
                raw_subtype = (
                    upload_subtype
                    or clean_row.get("upload_subtype")
                    or clean_row.get("evaluation_type")
                    or ""
                ).lower().strip()

                eval_mapping = {
                    "ia1": "IA-1",
                    "ia2": "IA-2",
                    "ia_avg": "IA-AVG",
                    "see": "SEE"
                }

                target_eval_type = eval_mapping.get(
                    raw_subtype,
                    "IA-1"
                )

                default_max_marks = (
                    100.0
                    if target_eval_type == "SEE"
                    else 50.0
                )

                row_synchronized = False

                # --------------------------------------------------
                # Process Course Slots 1..9
                # --------------------------------------------------
                for i in range(1, 10):

                    course_code = str(
                        clean_row.get(f"course_code{i}", "")
                    ).strip().upper()

                    if not course_code:
                        continue

                    if course_code in ("NA", "NULL"):
                        continue

                    marks_val = clean_row.get(f"marks{i}")
                    attendance_val = clean_row.get(
                        f"attendance_percentage{i}"
                    )

                    # ----------------------------------------------
                    # Validate Course
                    # ----------------------------------------------
                    course = db.query(
                        models.DbCourse
                    ).filter(
                        models.DbCourse.course_code == course_code
                    ).first()

                    if not course:
                        continue

                    # ----------------------------------------------
                    # Fetch/Create Registration
                    # ----------------------------------------------
                    registration = db.query(
                        models.DbCourseRegistration
                    ).filter(
                        models.DbCourseRegistration.student_id
                        == usn_or_id,
                        models.DbCourseRegistration.course_id
                        == course_code,
                        models.DbCourseRegistration.semester
                        == semester_int,
                        models.DbCourseRegistration.academic_year
                        == academic_year_val
                    ).first()

                    if not registration:

                        registration = models.DbCourseRegistration(
                            student_id=usn_or_id,
                            course_id=course_code,
                            semester=semester_int,
                            academic_year=academic_year_val
                        )

                        db.add(registration)
                        db.flush()

                    # ----------------------------------------------
                    # Fetch/Create Ledger
                    # ----------------------------------------------
                    ledger = db.query(
                        models.DbMarksLedger
                    ).filter(
                        models.DbMarksLedger.registration_id
                        == registration.id,
                        models.DbMarksLedger.evaluation_type
                        == target_eval_type
                    ).first()

                    if not ledger:

                        ledger = models.DbMarksLedger(
                            registration_id=registration.id,
                            evaluation_type=target_eval_type,
                            max_marks=default_max_marks
                        )

                        db.add(ledger)

                    else:
                        ledger.max_marks = default_max_marks

                    # ----------------------------------------------
                    # Marks
                    # ----------------------------------------------
                    if (
                        marks_val is not None
                        and str(marks_val).strip().upper() != "NA"
                        and str(marks_val).strip() != ""
                    ):
                        try:
                            ledger.marks_obtained = float(marks_val)
                            row_synchronized = True
                        except Exception:
                            pass

                    # ----------------------------------------------
                    # Attendance
                    # ----------------------------------------------
                    if (
                        attendance_val is not None
                        and str(attendance_val).strip().upper() != "NA"
                        and str(attendance_val).strip() != ""
                    ):
                        try:
                            ledger.attendance_percentage = float(
                                attendance_val
                            )
                            row_synchronized = True
                        except Exception:
                            pass

                if row_synchronized:
                    total_updated_records += 1
                db.commit()
                
            # elif upload_type == "academic":
            #     usn_or_id = clean_row.get("usn_or_id", clean_row.get("student_id", "")).strip()
            #     if not usn_or_id:
            #         continue

            #     # 1. Verification Security Guard & Department Boundary Enforcements
            #     student_user = db.query(models.DbUser).filter(
            #         models.DbUser.usn_or_id == usn_or_id, 
            #         models.DbUser.role == "student"
            #     ).first()
            #     if not student_user:
            #         continue
            #     if current_user.role != "superadmin" and student_user.department.lower() != current_user.department.lower():
            #         boundary_violations_skipped += 1
            #         continue

            #     # Extract lookup attributes directly from the CSV row layout
            #     semester_val = clean_row.get("semester")
            #     academic_year_val = clean_row.get("academic_year", "").strip()

            #     try:
            #         semester_int = int(float(str(semester_val).strip()))
            #     except (ValueError, TypeError):
            #         semester_int = None

            #     # Normalize frontend selection keys to match database Enum strings exactly
            #     eval_mapping = {
            #         "ia1": "IA-1",
            #         "ia2": "IA-2",
            #         "ia_avg": "IA-AVG",
            #         "see": "SEE"
            #     }
                
            #     # FIXED: The frontend form appends the modifier under the payload key "upload_type"
            #     # when mainSelector.value == "academic". Let's capture it accurately.
            #     if 'request' in locals() and hasattr(request, 'form'):
            #         raw_subtype = request.form.get("upload_type") or request.form.get("upload_subtype") or ""
            #     else:
            #         raw_subtype = clean_row.get("upload_type") or clean_row.get("upload_subtype") or ""

            #     target_eval_type = eval_mapping.get(str(raw_subtype).lower().strip(), "IA-1")

            #     # Dynamic Max Marks scaling constraint boundary logic rules
            #     default_max_marks = 100.0 if target_eval_type == "SEE" else 50.0

            #     # Tracks whether any actual course was processed and saved for this student row
            #     row_synchronized = False

            #     # 2. Dynamic 9-Slot Matrix Row Scanner
            #     for i in range(1, 10):
            #         course_key = f"course_code{i}"
            #         marks_key = f"marks{i}"
            #         att_key = f"attendance_percentage{i}"

            #         raw_course_code = clean_row.get(course_key, "")
            #         if not raw_course_code:
            #             continue
                        
            #         course_code = str(raw_course_code).strip().upper()
            #         # Gracefully skip empty parameters or unused template columns marked with 'NA'
            #         if course_code in ("", "NA", "NULL"):
            #             continue

            #         # =========================================================================
            #         # 3. CODE TO FETCH REGISTRATION_ID FROM COURSE_REGISTRATION TABLE
            #         # =========================================================================
            #         # Fetch the course record to obtain its primary key ID for registration lookup
            #         course_node = db.query(models.DbCourse).filter(models.DbCourse.course_code == course_code).first()
            #         if not course_node:
            #             # No such course in catalog, skip this entry
            #             continue
            #         reg_query = db.query(models.DbCourseRegistration).filter(
            #             models.DbCourseRegistration.student_id == usn_or_id,
            #             models.DbCourseRegistration.course_id == course_node.course_code
            #         )
                    
            #         # Apply semester and academic year filters as before
            #         if semester_int is not None:
            #             # Resilient matching supporting both Integer, explicit String, or Wildcard fields (e.g. "4th Sem")
            #             semester_str = str(semester_int)
            #             reg_query = reg_query.filter(
            #                 (models.DbCourseRegistration.semester == semester_int) | 
            #                 (models.DbCourseRegistration.semester == semester_str) |
            #                 (models.DbCourseRegistration.semester.like(f"%{semester_str}%"))
            #             )
            #         if academic_year_val:
            #             reg_query = reg_query.filter(models.DbCourseRegistration.academic_year == academic_year_val)

            #         registration = reg_query.first()
            #         if not registration:
            #             # Create new registration entry for this student-course pairing
            #             new_reg = models.DbCourseRegistration(
            #                 student_id=usn_or_id,
            #                 course_id=course_node.course_code,
            #                 semester=semester_int if semester_int is not None else 1,
            #                 academic_year=academic_year_val or ''
            #             )
            #             db.add(new_reg)
            #             db.flush()
            #             registration = new_reg
            #             row_synchronized = True
            #         # SUCCESS: Isolated registration database ID key
            #         fetched_registration_id = registration.id

            #         # =========================================================================
            #         # 4. UPSERT INTO DB_MARKS_LEDGER TABLE
            #         # =========================================================================
            #         ledger_record = db.query(models.DbMarksLedger).filter(
            #             models.DbMarksLedger.registration_id == fetched_registration_id,
            #             models.DbMarksLedger.evaluation_type == target_eval_type
            #         ).first()

            #         # Create a new atomic line item entry if it doesn't exist
            #         if not ledger_record:
            #             ledger_record = models.DbMarksLedger(
            #                 registration_id=fetched_registration_id,  # MAPPED MIGRATION TRACKING KEY
            #                 evaluation_type=target_eval_type,
            #                 max_marks=default_max_marks,
            #                 marks_obtained=None,      # Defaulting to schema spec None
            #                 attendance_percentage=None # Defaulting to schema spec None
            #             )
            #             db.add(ledger_record)
            #             db.flush() # Secure reference pointer instantly
            #         else:
            #             ledger_record.max_marks = default_max_marks

            #         # 5. Extract, sanitize, and persist numerical performance targets
            #         raw_marks = clean_row.get(marks_key)
            #         raw_att = clean_row.get(att_key)

            #         # Update evaluations marks payload metrics
            #         if raw_marks is not None and str(raw_marks).strip().upper() != "NA":
            #             try:
            #                 ledger_record.marks_obtained = float(str(raw_marks).strip())
            #                 row_synchronized = True
            #             except (ValueError, TypeError):
            #                 pass

            #         # Update running attendance percentages
            #         if raw_att is not None and str(raw_att).strip().upper() != "NA":
            #             try:
            #                 ledger_record.attendance_percentage = float(str(raw_att).strip())
            #                 row_synchronized = True
            #             except (ValueError, TypeError):
            #                 pass

            #     # Commit changes directly to the database only if data updates were made
            #     if row_synchronized:
            #         db.commit()
            #         total_updated_records += 1
                    
            # CASE 3: Comprehensive Faculty Mentor & Professional Tracking Profile Ingestion
            elif upload_type == "mentor_data":
                email = str(clean_row.get("email", "")).strip()
                if not email:
                    continue

                employee_id = str(clean_row.get("usn_or_id", "")).strip()
                if not employee_id:
                    continue  # Ensure key mapping reference string is valid

                # 1. UPSERT Operational Authenticator Target Block in the Users Registry
                mentor_node = db.query(models.DbUser).filter(models.DbUser.email == email).first()
                if not mentor_node:
                    mentor_node = models.DbUser(
                        email=email,
                        hashed_password=hash_password(clean_row.get("password", "FacultyShield99$")),
                        role="mentor",
                        display_name=clean_row.get("display_name", "Faculty Advisor"),
                        department=row_dept if row_dept else current_user.department,
                        usn_or_id=employee_id,
                        is_active=True
                    )
                    db.add(mentor_node)
                    db.flush()  # Extract structural reference state updates cleanly
                else:
                    mentor_node.display_name = clean_row.get("display_name", mentor_node.display_name)
                    mentor_node.usn_or_id = employee_id
                    if row_dept:
                        mentor_node.department = row_dept
                    db.flush()

                # 2. Extract and Parse Multi-Format Tenure Timeline Timestamps safely
                raw_doj = clean_row.get("date_of_joining", "").strip()
                try:
                    # Accepts 'YYYY-MM-DD HH:MM:SS' or fallbacks to plain 'YYYY-MM-DD'
                    doj_parsed = datetime.strptime(raw_doj, "%Y-%m-%d %H:%M:%S") if " " in raw_doj else datetime.strptime(raw_doj, "%Y-%m-%d")
                except (ValueError, TypeError):
                    doj_parsed = datetime.now() # Production defensive schema fallback marker

                raw_dor = clean_row.get("relieving_date", "").strip()
                dor_parsed = None
                if raw_dor and raw_dor.lower() != "na":
                    try:
                        dor_parsed = datetime.strptime(raw_dor, "%Y-%m-%d %H:%M:%S") if " " in raw_dor else datetime.strptime(raw_dor, "%Y-%m-%d")
                    except (ValueError, TypeError):
                        dor_parsed = None

                # 3. Defensive Floating Point Quantifier Normalizer Utility Closure
                def parse_exp_metric(field_key: str) -> float:
                    val = clean_row.get(field_key, "0.0").strip()
                    if not val or val.lower() == "na":
                        return 0.0
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        return 0.0

                exp_college = int(parse_exp_metric("experience_in_college"))
                exp_teach = parse_exp_metric("teaching_experience_years")
                exp_research = parse_exp_metric("research_experience_years")
                exp_industry = parse_exp_metric("industry_experience_years")
                exp_other = parse_exp_metric("other_experience_years")
                exp_total = parse_exp_metric("total_work_experience_years")

                # If total experience column isn't provided or is zero, auto-calculate it dynamically
                if exp_total == 0.0:
                    exp_total = round(exp_teach + exp_research + exp_industry + exp_other, 2)

                # 4. UPSERT Comprehensive Relational Background Detail Record Block
                mentor_profile = db.query(models.DbMentorDetail).filter(models.DbMentorDetail.employee_id == employee_id).first()
                
                if not mentor_profile:
                    new_profile = models.DbMentorDetail(
                        employee_id=employee_id,
                        name=clean_row.get("display_name", "Faculty Advisor"),
                        email=email,
                        phone_number=clean_row.get("phone_number") or None,
                        department_name=row_dept if row_dept else (current_user.department if current_user else "Operations"),
                        designation=clean_row.get("designation", "Assistant Professor"),
                        gender=clean_row.get("gender") or None,
                        highest_qualification=clean_row.get("highest_qualification") or None,
                        pan_card_number=clean_row.get("pan_card_number") or None,
                        aadhaar_card_number=clean_row.get("aadhaar_card_number") or None,
                        date_of_joining=doj_parsed,
                        relieving_date=dor_parsed,
                        experience_in_college=exp_college,
                        teaching_experience_years=exp_teach,
                        research_experience_years=exp_research,
                        industry_experience_years=exp_industry,
                        other_experience_years=exp_other,
                        total_work_experience_years=exp_total
                    )
                    db.add(new_profile)
                else:
                    # Update Existing Extended Structural Metrics cleanly on re-upload routines
                    mentor_profile.name = clean_row.get("display_name", mentor_profile.name)
                    mentor_profile.email = email
                    mentor_profile.phone_number = clean_row.get("phone_number", mentor_profile.phone_number)
                    mentor_profile.designation = clean_row.get("designation", mentor_profile.designation)
                    mentor_profile.highest_qualification = clean_row.get("highest_qualification", mentor_profile.highest_qualification)
                    mentor_profile.pan_card_number = clean_row.get("pan_card_number", mentor_profile.pan_card_number)
                    mentor_profile.aadhaar_card_number = clean_row.get("aadhaar_card_number", mentor_profile.aadhaar_card_number)
                    mentor_profile.date_of_joining = doj_parsed
                    mentor_profile.relieving_date = dor_parsed
                    mentor_profile.experience_in_college = exp_college
                    mentor_profile.teaching_experience_years = exp_teach
                    mentor_profile.research_experience_years = exp_research
                    mentor_profile.industry_experience_years = exp_industry
                    mentor_profile.other_experience_years = exp_other
                    mentor_profile.total_work_experience_years = exp_total

                total_updated_records += 1

            # CASE 4: Master Course Catalogue Storage Index
            elif upload_type == "course_catalog":
                course_code = clean_row.get("course_code", "").upper()
                course_title = clean_row.get("course_name", clean_row.get("course_title", ""))
                
                if not course_code or not course_title:
                    continue

                course_node = db.query(models.DbCourse).filter(models.DbCourse.course_code == course_code).first()
                if not course_node:
                    course_node = models.DbCourse(
                        course_code=course_code,
                        course_name=course_title,
                        assigned_semester=int(clean_row.get("assigned_semester") or 1),
                        department=row_dept if row_dept else current_user.department
                    )
                    db.add(course_node)
                else:
                    course_node.course_name = course_title
                    course_node.assigned_semester = int(clean_row.get("assigned_semester") or course_node.assigned_semester or 1)
                
                total_updated_records += 1

            # CASE 5: Registration Association Bridging Layout (Multi-Course Pivot Array Loop)
            elif upload_type == "course_registration":
                student_id = clean_row.get("student_id", clean_row.get("usn_or_id", "")).strip()
                if not student_id:
                    continue

                # 1. Verify that the target student exists and enforce department boundaries
                student_user = db.query(models.DbUser).filter(models.DbUser.usn_or_id == student_id).first()
                if not student_user:
                    continue
                if current_user.role != "superadmin" and student_user.department.lower() != current_user.department.lower():
                    boundary_violations_skipped += 1
                    continue

                # Extract standard metadata values shared across registrations for this row
                try:
                    semester_val = int(float(str(clean_row.get("semester", "1")).strip()))
                except (ValueError, TypeError):
                    semester_val = 1
                
                academic_year_val = clean_row.get("academic_year", "2025-2026").strip()

                # 2. Iterate dynamically over the 9 course columns provided in your template layout
                for i in range(1, 10):
                    course_key = f"course_code{i}"
                    raw_course_code = clean_row.get(course_key, "")
                    
                    if not raw_course_code:
                        continue
                        
                    course_code = raw_course_code.strip().upper()

                    # Explicit Requirement: Skip "NA" values and proceed to next course code index
                    if course_code == "NA" or course_code == "":
                        continue

                    # 3. Fetch the true course metadata matching the code string inside the catalog matrix   
                    course_node = db.query(models.DbCourse).filter(models.DbCourse.course_code == course_code).first()
                    if not course_node:
                        continue  # Skip unmapped code values seamlessly

                    # 4. Verify uniqueness using the business code string to avoid duplicate errors
                    existing_reg = db.query(models.DbCourseRegistration).filter(
                        models.DbCourseRegistration.student_id == student_id,
                        models.DbCourseRegistration.course_id == course_node.course_code, # String Comparison
                        models.DbCourseRegistration.academic_year == academic_year_val,
                        models.DbCourseRegistration.semester == semester_val
                    ).first()

                    if not existing_reg:
                        # 5. Commit record using the string alphanumeric business identifier code
                        new_reg = models.DbCourseRegistration(
                            student_id=student_id,
                            course_id=course_node.course_code,  # FIXED: Extracted course_code string instead of integer .id
                            semester=semester_val,
                            academic_year=academic_year_val
                        )
                        db.add(new_reg)
                
                # Increment the bulk tracker variable to count row synchronization updates
                total_updated_records += 1

        db.commit()

        msg_payload = f"Structural collection processed successfully. Synchronized {total_updated_records} entries."
        if boundary_violations_skipped > 0:
            msg_payload += f" Protective routines skipped {boundary_violations_skipped} cross-department layout attempts."

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "msg": msg_payload,
                "updated": total_updated_records
            }
        )

    except Exception as matrix_err:
        db.rollback()
        return JSONResponse(
            status_code=400,
            content={
                "success": False, 
                "msg": f"PostgreSQL Ledger Ingestion Error: {str(matrix_err)}"
            }
        )
# =========================================================================    
#   COHORT RELATIONSHIPS & CONFIG CALIBRATIONS
# =========================================================================
from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, DateTime
# Make sure Boolean is imported from sqlalchemy above!

@app.post("/admin/relationships/map")
async def link_cohort_student_to_mentor(
    student_id: int = Form(...),
    mentor_id: int = Form(None),              
    deallocate: bool = Form(False),            
    db: Session = Depends(get_db),
    request: Request = None
):
    """Updates or safely clears (deallocates) mentor advisory mappings with strict multi-tenant scope validation."""
    user_token = request.cookies.get("session_token")
    current_admin = db.query(models.DbUser).filter(models.DbUser.email == user_token).first()
    
    if not current_admin or current_admin.role not in ["admin", "superadmin"]:
        return RedirectResponse(url="/auth/login/admin?err=Unauthorized", status_code=303)
    
    try:
        student_user = db.query(models.DbUser).filter(models.DbUser.id == student_id, models.DbUser.role == "student").first()
        if not student_user:
            return RedirectResponse(url="/dashboard/admin?err=Target+student+identity+node+not+found.", status_code=303)
            
        if current_admin.role != "superadmin" and student_user.department != current_admin.department:
             return RedirectResponse(url="/dashboard/admin?err=Access+Denied:+Student+belongs+to+another+branch.", status_code=303)
        
        if deallocate or mentor_id is None:
            student_user.mentor_id = None
            msg_text = "Cohort+route+link+successfully+deallocated."
        else:
            mentor_user = db.query(models.DbUser).filter(models.DbUser.id == mentor_id, models.DbUser.role == "mentor").first()
            if not mentor_user:
                return RedirectResponse(url="/dashboard/admin?err=Target+faculty+mentor+not+found.", status_code=303)
                
            if current_admin.role != "superadmin" and mentor_user.department != current_admin.department:
                return RedirectResponse(url="/dashboard/admin?err=Access+Denied:+Mentor+belongs+to+another+branch.", status_code=303)
                
            student_user.mentor_id = mentor_id
            msg_text = "Cohort+route+link+bound+successfully."
            
        db.commit()
        return RedirectResponse(url=f"/dashboard/admin?tab=role_enforcement&msg={msg_text}", status_code=303)
        
    except Exception as e:
        db.rollback()
        return RedirectResponse(url=f"/dashboard/admin?err=Mapping+Operation+Fault:+{str(e)}", status_code=303)


@app.post("/admin/config/assign-mentor")
async def process_mentor_linkage(
    request: Request,
    student_usns: List[str] = Form([]),
    mentor_id_code: str = Form(None), 
    deallocate: bool = Form(False),
    db: Session = Depends(get_db)
):
    """
    Binds or unbinds batch student tracking profiles to mentor advisory records, 
    updating records dynamically based on active metadata entries in AcademicTermControl.
    """
    # 1. AUTHENTICATION & ACCESS BOUNDARY CHECK (FIXED FOR CRYPTO TOKENS)
    user_token = request.cookies.get("session_token")
    if not user_token:
        return RedirectResponse(url="/auth/login/admin?err=Session+Expired", status_code=303)
        
    try:
        # Decrypt/verify token payload to pull the underlying integer User ID
        decrypted_user_id = verify_session_token(user_token)
        current_admin = db.query(models.DbUser).filter(models.DbUser.email == user_token).first()
    except Exception:
        current_admin = None

    if not current_admin or current_admin.role not in ["admin", "superadmin"]:
        return RedirectResponse(url="/auth/login/admin?err=Unauthorized", status_code=303)

    if not student_usns:
        return RedirectResponse(url="/dashboard/admin?action=assign_mentor&err=No+students+selected+from+ledger.", status_code=303)

    # 2. RESOLVE COHORT METADATA (DYNAMIC DATABASE FETCH)
    active_academic_year, active_semester_type = get_active_mapping_term(db, current_admin)

    # 3. MENTOR RESOLUTION
    mentor = None
    if not deallocate:
        if not mentor_id_code:
            return RedirectResponse(url="/dashboard/admin?action=assign_mentor&err=Please+select+a+target+mentor.", status_code=303)
        
        mentor = db.query(models.DbUser).filter(
            models.DbUser.usn_or_id == mentor_id_code, 
            models.DbUser.role == "mentor",
        ).first()
        
        if not mentor:
            return RedirectResponse(url="/dashboard/admin?action=assign_mentor&err=Targeted+Faculty+Mentor+not+found.", status_code=303)

    # 4. ITERATE TARGET STUDENT MATRIX SELECTION ITEMS
    for usn in student_usns:
        student = db.query(models.DbUser).filter(
            models.DbUser.usn_or_id == usn, 
            models.DbUser.role == "student"
        ).first()
        
        if not student:
            continue

        # 5. MULTI-TENANT DEPARTMENT SECURITY SANITIZATION
        if current_admin.role != "superadmin":
            if student.department != current_admin.department:
                continue 
            if mentor and mentor.department != current_admin.department:
                continue

        if deallocate:
          if deallocate:
            # ==========================================================
            # DEALLOCATION ROUTINE WORKFLOW (DELETE + NULL CACHE)
            # ==========================================================
            # 1. Update the base reference state column on 'users' table to NULL
            student.mentor_id = None
            
            # 2. Completely delete the row from the mapping table ledger
            db.query(models.MentorStudentMapping).filter(
                models.MentorStudentMapping.student_usn_or_id == student.usn_or_id,
                models.MentorStudentMapping.academic_year == active_academic_year,
                models.MentorStudentMapping.semester_type == active_semester_type
            ).delete(synchronize_session=False)
            
        else:
            # ==========================================
            # ALLOCATION / UPDATE ROUTINE WORKFLOW
            # ==========================================
            # 1. Update the base user profile reference field (uses user.id)
            student.mentor_id = mentor.id
            
            # 2. FETCH THE ALIAS TRACKING METADATA PROFILE USING THE UNIQUE KEY RELATIONSHIP
            # FIXED: Changed models.MentorDetail to models.DbMentorDetail
            mentor_profile = db.query(models.DbMentorDetail).filter(
                models.DbMentorDetail.employee_id == mentor_id_code
            ).first()
            
            if not mentor_profile:
                return RedirectResponse(
                    url=f"/dashboard/admin?action=assign_mentor&err=Profile+not+found+in+mentor_details+for+{mentor.display_name}.", 
                    status_code=303
                )
                
            # Ensure the mentor profile has a valid unique identifier set up
            if not mentor_profile.employee_id:
                return RedirectResponse(
                    url=f"/dashboard/admin?action=assign_mentor&err=Mentor+is+missing+employee_id+string+reference.", 
                    status_code=303
                )

            # 3. Check for an existing mapping record matching this term
            existing_mapping = db.query(models.MentorStudentMapping).filter(
                models.MentorStudentMapping.student_usn_or_id == student.usn_or_id,
                models.MentorStudentMapping.academic_year == active_academic_year,
                models.MentorStudentMapping.semester_type == active_semester_type
            ).first()
            
            if existing_mapping:
                # Assign the VARCHAR employee_id string to meet the constraint rule
                existing_mapping.mentor_id = mentor_profile.employee_id
                existing_mapping.is_active = True
            else:
                # Insert using the VARCHAR employee_id string
                new_mapping = models.MentorStudentMapping(
                    mentor_id=mentor_profile.employee_id, 
                    student_usn_or_id=student.usn_or_id,
                    academic_year=active_academic_year,
                    semester_type=active_semester_type,
                    is_active=True
                )
                db.add(new_mapping)

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        return RedirectResponse(url=f"/dashboard/admin?action=assign_mentor&err=Transaction+Failed:+{str(e)}", status_code=303)

    return RedirectResponse(
        url="/dashboard/admin?action=assign_mentor&msg=Operational+ledger+and+relational+mappings+updated+successfully.", 
        status_code=303
    )

@app.post("/admin/config/calibrate")
async def calibrate_thresholds(
    attendance_threshold: float = Form(...),
    cgpa_threshold: float = Form(...),
    term_name: str = Form(...),
    db: Session = Depends(get_db),
    request: Request = None
):
    user_token = request.cookies.get("session_token")
    current_admin = db.query(models.DbUser).filter(models.DbUser.email == user_token).first()
    
    target_dept = "Global" if current_admin.role == "superadmin" else current_admin.department 
    config = db.query(models.SystemConfiguration).filter(models.SystemConfiguration.department == target_dept).first()
    if not config:
        config = models.SystemConfiguration(department=target_dept)
        db.add(config)
        
    config.attendance_threshold = attendance_threshold
    config.cgpa_threshold = cgpa_threshold
    config.term_name = term_name
    db.commit()
    return RedirectResponse(url="/dashboard/admin?tab=overview&msg=Thresholds+updated+for+" + target_dept, status_code=303)


@app.post("/admin/config/transition-term")
async def execute_term_transition_reset(next_term_name: str = Form(...), confirm_archive: bool = Form(False), request: Request = None, db: Session = Depends(get_db)):
    """Term Control: Transitioning academic semesters (Superadmin restricted exclusively)."""
    user_token = request.cookies.get("session_token")
    current_admin = db.query(models.DbUser).filter(models.DbUser.email == user_token).first()
    
    if not current_admin or current_admin.role != "superadmin":
         return RedirectResponse(url="/dashboard/admin?err=Access+Denied:+Only+the+superadmin can invoke master semester wipes.", status_code=303)

    if not confirm_archive:
        return RedirectResponse(url="/dashboard/admin?err=You+must+confirm+historical+archival.", status_code=303)
    return RedirectResponse(url=f"/dashboard/admin?msg=Transitioned+to+{next_term_name}+successfully.", status_code=303)


# =========================================================================
#      SECURE MULTI-TENANT EXPORT REPORT GENERATION VECTORS
# =========================================================================

@app.get("/admin/reports/academic-summary")
async def export_academic_report(
    request: Request, 
    db: Session = Depends(get_db), 
    current_admin: models.DbUser = Depends(get_current_active_admin)
):
    """Generates a CSV download sheet of students' academic profiles filtered by tenant boundaries."""
    query = db.query(
        models.DbUser.usn_or_id, models.DbUser.display_name, models.DbUser.department,
        models.DbStudentInfo.father_name, models.DbStudentInfo.phone_number,
        models.DbStudentInfo.tenth_percentage, models.DbStudentInfo.twelfth_percentage
    ).join(models.DbStudentInfo, models.DbUser.id == models.DbStudentInfo.student_id)

    if current_admin.role != "superadmin":
        query = query.filter(models.DbUser.department == current_admin.department)

    records = query.all()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["USN / ID", "Student Name", "Department", "Father Name", "Contact Phone", "10th %", "12th %"])
    
    for r in records:
        writer.writerow([r.usn_or_id, r.display_name, r.department, r.father_name, r.phone_number, r.tenth_percentage, r.twelfth_percentage])
        
    output.seek(0)
    filename = f"Academic_Report_{current_admin.department if current_admin.role == 'admin' else 'Global'}.csv"
    return StreamingResponse(io.BytesIO(output.getvalue().encode("utf-8")), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={filename}"})


@app.get("/admin/export/risk-csv")
@app.get("/admin/reports/performance-risk")
async def export_risk_report(
    request: Request, 
    db: Session = Depends(get_db), 
    current_admin: models.DbUser = Depends(get_current_active_admin)
):
    """Generates an early-warning risk report isolating structural threshold anomalies."""
    students_query = db.query(models.DbUser).filter(
        models.DbUser.role == "student",
        models.DbUser.is_active == True
    )
    if current_admin.role != "superadmin":
        students_query = students_query.filter(models.DbUser.department == current_admin.department)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["USN / ID", "Student Name", "Department", "Semester", "CGPA", "Attendance %", "Active Backlogs"])
    
    for student in students_query.all():
        semester = ""
        attendance_val = 100.0
        cgpa_val = 10.0
        backlogs_count = 0

        current_record = db.query(models.DbAcademicRecord).filter(
            models.DbAcademicRecord.student_id == student.usn_or_id,
            models.DbAcademicRecord.is_historical_snapshot == False
        ).order_by(models.DbAcademicRecord.semester.desc()).first()

        if current_record:
            semester = current_record.semester
            attendance_val = current_record.overall_attendance if current_record.overall_attendance is not None else 100.0
            cgpa_val = current_record.cumulative_cgpa if current_record.cumulative_cgpa is not None else 10.0
            backlogs_count = current_record.backlogs_count or 0
        else:
            latest_registration = db.query(models.DbCourseRegistration).filter(
                models.DbCourseRegistration.student_id == student.usn_or_id
            ).order_by(
                models.DbCourseRegistration.academic_year.desc(),
                models.DbCourseRegistration.semester.desc()
            ).first()

            if latest_registration:
                semester = latest_registration.semester
                latest_registrations = db.query(models.DbCourseRegistration).filter(
                    models.DbCourseRegistration.student_id == student.usn_or_id,
                    models.DbCourseRegistration.academic_year == latest_registration.academic_year,
                    models.DbCourseRegistration.semester == latest_registration.semester
                ).all()
                latest_registration_ids = [reg.id for reg in latest_registrations]

                if latest_registration_ids:
                    marks_rows = db.query(models.DbMarksLedger).filter(
                        models.DbMarksLedger.registration_id.in_(latest_registration_ids)
                    ).all()

                    attendance_samples = [
                        row.attendance_percentage
                        for row in marks_rows
                        if row.attendance_percentage is not None
                    ]
                    if attendance_samples:
                        attendance_val = round(sum(attendance_samples) / len(attendance_samples), 2)

                    score_samples = [
                        (row.marks_obtained / row.max_marks) * 10
                        for row in marks_rows
                        if row.marks_obtained is not None and row.max_marks
                    ]
                    if score_samples:
                        cgpa_val = round(sum(score_samples) / len(score_samples), 2)
                        backlogs_count = sum(1 for score in score_samples if score < 4.0)

        if attendance_val < 75.0 or cgpa_val < 6.5 or backlogs_count > 0:
            writer.writerow([
                student.usn_or_id,
                student.display_name,
                student.department,
                semester,
                cgpa_val,
                attendance_val,
                backlogs_count
            ])
        
    output.seek(0)
    filename = f"Risk_Analysis_Report_{current_admin.department if current_admin.role == 'admin' else 'Global'}.csv"
    return StreamingResponse(io.BytesIO(output.getvalue().encode("utf-8")), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={filename}"})


# =========================================================================
#        ADMINISTRATIVE ORCHESTRATION & ANALYTICS WORKSPACE
# =========================================================================

@app.get("/dashboard/admin", response_class=HTMLResponse)
async def render_admin_dashboard(
    request: Request,
    msg: str = None,
    err: str = None,
    db: Session = Depends(get_db)
):
    """Renders administrative orchestrator panels securely handling stats matrix builders."""
    user_token = request.cookies.get("session_token")
    if not user_token:
        return RedirectResponse(url="/auth/login/admin?err=Session+Expired", status_code=303)
        
    current_admin = db.query(models.DbUser).filter(models.DbUser.email == user_token).first()
    if not current_admin or current_admin.role not in ["admin", "superadmin"]:
        return RedirectResponse(url="/auth/login/admin?err=Unauthorized", status_code=303)

    selected_branch = (request.query_params.get("risk_branch") or "").strip()
    selected_semester = (request.query_params.get("risk_semester") or "").strip()
    selected_sort = (request.query_params.get("risk_sort") or "status").strip()
    selected_limit_raw = (request.query_params.get("risk_limit") or "10").strip()
    selected_page_raw = (request.query_params.get("risk_page") or "1").strip()

    allowed_limits = {10, 50, 100}
    try:
        selected_limit = int(selected_limit_raw)
    except ValueError:
        selected_limit = 10
    if selected_limit not in allowed_limits:
        selected_limit = 10

    allowed_sorts = {"cgpa", "attendance", "status"}
    if selected_sort not in allowed_sorts:
        selected_sort = "status"
    try:
        selected_page = int(selected_page_raw)
    except ValueError:
        selected_page = 1
    if selected_page < 1:
        selected_page = 1

    # Establish role boundaries
    students_query = db.query(models.DbUser).filter(
        models.DbUser.role == "student", 
        models.DbUser.is_active == True
    ).options(joinedload(models.DbUser.academic_state))
    mentors_query = db.query(models.DbUser).filter(models.DbUser.role == "mentor", models.DbUser.is_active == True)
    counselors_query = db.query(models.DbUser).filter(models.DbUser.role == "counselor", models.DbUser.is_active == True)
    
    if current_admin.role != "superadmin":
        students_query = students_query.filter(models.DbUser.department == current_admin.department)
        mentors_query = mentors_query.filter(models.DbUser.department == current_admin.department)
        counselors_query = counselors_query.filter(models.DbUser.department == current_admin.department)

    branch_options = [
        row[0] for row in students_query.with_entities(models.DbUser.department)
        .filter(models.DbUser.department.isnot(None))
        .distinct()
        .order_by(models.DbUser.department.asc())
        .all()
        if row[0]
    ]
    if current_admin.role != "superadmin":
        selected_branch = current_admin.department or ""
    elif selected_branch:
        students_query = students_query.filter(models.DbUser.department == selected_branch)

    total_students = students_query.count()
    total_mentors = mentors_query.count()
    total_counselors = counselors_query.count()

    all_students_raw = students_query.all()
    all_mentors_raw = mentors_query.all()

    active_academic_year, active_semester_type = get_active_mapping_term(db, current_admin)
    active_mappings = db.query(models.MentorStudentMapping).filter(
        models.MentorStudentMapping.academic_year == active_academic_year,
        models.MentorStudentMapping.semester_type == active_semester_type,
        models.MentorStudentMapping.is_active == True
    ).all()
    mapping_by_student = {mapping.student_usn_or_id: mapping for mapping in active_mappings}

    # mentor_map = {m.id: m.display_name for m in all_mentors_raw}
    # mentor_code_map = {m.id: m.usn_or_id for m in all_mentors_raw}
    mentor_map = {m.usn_or_id: m.display_name for m in all_mentors_raw if m.usn_or_id}
    mentor_code_map = {m.usn_or_id: m.usn_or_id for m in all_mentors_raw if m.usn_or_id}
    assignment_counts_raw = db.query(
        models.MentorStudentMapping.mentor_id,
        func.count(models.MentorStudentMapping.id).label("total_assigned")
    ).filter(
        models.MentorStudentMapping.academic_year == active_academic_year,
        models.MentorStudentMapping.semester_type == active_semester_type,
        models.MentorStudentMapping.is_active == True
    ).group_by(models.MentorStudentMapping.mentor_id).all()

    # Convert the grouped result row sequence into an easily readable dictionary
    # Key: mentor_id string (employee_id) -> Value: Integer count of active mappings
    mentor_count_map = {row.mentor_id: row.total_assigned for row in assignment_counts_raw}

    # 2. Inject the assigned_count parameter into the mentor list dictionary items
    all_mentors_list = [
        {
            "id": m.id, 
            "usn_or_id": m.usn_or_id, 
            "display_name": m.display_name, 
            "department": m.department,
            # Fall back to 0 if the mentor does not have an active structural mapping row record yet
            "assigned_count": mentor_count_map.get(m.usn_or_id, 0) 
        } for m in all_mentors_raw
    ]
    #all_mentors_list = [{"id": m.id, "usn_or_id": m.usn_or_id, "display_name": m.display_name, "department": m.department} for m in all_mentors_raw]
    all_students_list = [
        {
            "id": s.id, "usn_or_id": s.usn_or_id, "display_name": s.display_name, "department": s.department,
            "mentor_name": mentor_map.get(mapping_by_student[s.usn_or_id].mentor_id, None) if s.usn_or_id in mapping_by_student else None,
            "mentor_id_code": mentor_code_map.get(mapping_by_student[s.usn_or_id].mentor_id, None) if s.usn_or_id in mapping_by_student else None,
            "mapping_term": f"{active_academic_year} ({active_semester_type})" if s.usn_or_id in mapping_by_student else None,
            "current_semester": s.academic_state.current_semester if s.academic_state else None
        } for s in all_students_raw
    ]
    
    counseling_roster = []
    dept_critical_counts = {}

    for student in all_students_raw:
        attendance_val = 100.0
        cgpa_val = 10.0
        has_backlogs = False
        semester_val = ""

        # FIX: Query academic_records using student's alphanumeric tracking code (usn_or_id) 
        # and filter for the active snapshot record node
        academic_record_query = db.query(models.DbAcademicRecord).filter(
            models.DbAcademicRecord.student_id == student.usn_or_id,
            models.DbAcademicRecord.is_historical_snapshot == False
        )
        if selected_semester:
            try:
                academic_record_query = academic_record_query.filter(models.DbAcademicRecord.semester == int(selected_semester))
            except ValueError:
                academic_record_query = academic_record_query.filter(models.DbAcademicRecord.semester == -1)

        current_record = academic_record_query.order_by(models.DbAcademicRecord.semester.desc()).first()

        if current_record:
            semester_val = current_record.semester
            attendance_val = current_record.overall_attendance if current_record.overall_attendance is not None else 100.0
            cgpa_val = current_record.cumulative_cgpa if current_record.cumulative_cgpa is not None else 10.0
            has_backlogs = (current_record.backlogs_count > 0) if current_record.backlogs_count else False
        else:
            registration_query = db.query(models.DbCourseRegistration).filter(
                models.DbCourseRegistration.student_id == student.usn_or_id
            )
            if selected_semester:
                try:
                    registration_query = registration_query.filter(models.DbCourseRegistration.semester == int(selected_semester))
                except ValueError:
                    registration_query = registration_query.filter(models.DbCourseRegistration.semester == -1)

            latest_registration = registration_query.order_by(
                models.DbCourseRegistration.academic_year.desc(),
                models.DbCourseRegistration.semester.desc()
            ).first()

            if latest_registration:
                semester_val = latest_registration.semester
                latest_registrations = db.query(models.DbCourseRegistration).filter(
                    models.DbCourseRegistration.student_id == student.usn_or_id,
                    models.DbCourseRegistration.academic_year == latest_registration.academic_year,
                    models.DbCourseRegistration.semester == latest_registration.semester
                ).all()
                latest_registration_ids = [reg.id for reg in latest_registrations]

                if latest_registration_ids:
                    marks_rows = db.query(models.DbMarksLedger).filter(
                        models.DbMarksLedger.registration_id.in_(latest_registration_ids)
                    ).all()

                    attendance_samples = [
                        row.attendance_percentage
                        for row in marks_rows
                        if row.attendance_percentage is not None
                    ]
                    if attendance_samples:
                        attendance_val = round(sum(attendance_samples) / len(attendance_samples), 2)

                    score_samples = [
                        (row.marks_obtained / row.max_marks) * 10
                        for row in marks_rows
                        if row.marks_obtained is not None and row.max_marks
                    ]
                    if score_samples:
                        cgpa_val = round(sum(score_samples) / len(score_samples), 2)
                        has_backlogs = any(score < 4.0 for score in score_samples)

        is_critical = (attendance_val < 75.0 or cgpa_val < 6.5 or has_backlogs)
        
        # Lookups referencing core user relational keys remain tied to the internal ID sequence
        referred_to_counselor = db.query(models.CounselingRecord).filter(models.CounselingRecord.student_id == student.id).first()

        if referred_to_counselor or is_critical:
            status_tag = "Stable Progress"
            color_class = "teal"
            status_rank = 0
            if attendance_val < 70.0 or cgpa_val < 5.5:
                status_tag = "Critical Escalation"
                color_class = "rose"
                status_rank = 2
            elif is_critical:
                status_tag = "Under Review Warning"
                color_class = "amber"
                status_rank = 1

            counseling_roster.append({
                "name": student.display_name, "usn_or_id": student.usn_or_id, "department": student.department,
                "semester": semester_val or "N/A", "cgpa": cgpa_val, "attendance": f"{attendance_val}%",
                "attendance_value": attendance_val, "status": status_tag, "status_rank": status_rank, "color": color_class
            })

        if is_critical:
            dept_key = student.department or "General Operations"
            dept_critical_counts[dept_key] = dept_critical_counts.get(dept_key, 0) + 1

    sort_key_map = {
        "cgpa": lambda record: (record["cgpa"], record["attendance_value"]),
        "attendance": lambda record: (record["attendance_value"], record["cgpa"]),
        "status": lambda record: (-record["status_rank"], record["attendance_value"], record["cgpa"])
    }
    counseling_roster = sorted(counseling_roster, key=sort_key_map[selected_sort])
    roster_total = len(counseling_roster)
    total_pages = max((roster_total + selected_limit - 1) // selected_limit, 1)
    if selected_page > total_pages:
        selected_page = total_pages

    roster_start_index = (selected_page - 1) * selected_limit
    roster_end_index = roster_start_index + selected_limit
    counseling_roster = counseling_roster[roster_start_index:roster_end_index]
    pagination_base_params = urlencode({
        "risk_branch": selected_branch,
        "risk_semester": selected_semester,
        "risk_limit": selected_limit,
        "risk_sort": selected_sort
    })

    chart_analytics_data = [{"department": d, "count": c, "bar_width": min(c * 10, 100)} for d, c in dept_critical_counts.items()]

    semester_options = [
        row[0] for row in db.query(models.DbCourseRegistration.semester)
        .filter(models.DbCourseRegistration.semester.isnot(None))
        .distinct()
        .order_by(models.DbCourseRegistration.semester.asc())
        .all()
    ]

    # FIX: Query using the updated configuration registry mapping standard
    current_term = f"{active_academic_year} ({active_semester_type})"

    logs_list = []
    if current_admin.role == "superadmin":
        # FIX: Match updated properties from the audit_logs schema layout mutation
        audit_logs = db.query(models.AuditLog).order_by(models.AuditLog.executed_at.desc()).limit(50).all()
        logs_list = [
            {
                "timestamp": log.executed_at.strftime("%Y-%m-%d %H:%M:%S") if log.executed_at else "N/A",
                "event_type": log.event_type, 
                "severity": log.severity, 
                "details": f"[{log.operator_email}] - {log.details}", 
                "ip_address": log.ip_address
            } for log in audit_logs
        ]

    admin_payload = {
        "role": current_admin.role, "department": current_admin.department,
        "total_students": total_students, "total_mentors": total_mentors, "total_counselors": total_counselors,
        "current_term": current_term, "all_students_list": all_students_list, "all_mentors_list": all_mentors_list,
        "counseling_roster": counseling_roster, "chart_analytics": chart_analytics_data, "logs": logs_list,
        "branch_options": branch_options, "semester_options": semester_options,
        "risk_filters": {
            "branch": selected_branch,
            "semester": selected_semester,
            "limit": selected_limit,
            "sort": selected_sort,
            "page": selected_page
        },
        "risk_pagination": {
            "total": roster_total,
            "page": selected_page,
            "total_pages": total_pages,
            "start": roster_start_index + 1 if roster_total else 0,
            "end": min(roster_end_index, roster_total),
            "has_previous": selected_page > 1,
            "has_next": selected_page < total_pages,
            "previous_url": f"/dashboard/admin?{pagination_base_params}&risk_page={selected_page - 1}",
            "next_url": f"/dashboard/admin?{pagination_base_params}&risk_page={selected_page + 1}"
        }
    }
    students = db.query(models.DbUser).filter(
            models.DbUser.role == "student"
        ).options(
            # ✅ FIX: Change models.DbUser.mentor to models.DbUser.assigned_mentor
            joinedload(models.DbUser.assigned_mentor)
        ).all()

    filters = admin_payload["risk_filters"]

    return templates.TemplateResponse(request=request, name="admin.html", context={
        "request": request,
        "data": admin_payload,
        "msg": msg,
        "err": err,
        "current_user": current_admin,
        "faculty_list": all_mentors_list,
        "student_list": all_students_list,
        "filters": filters
    })

# =========================================================================
#        FALLBACK EXCEPTION ROUTER MIDDLEWARES
# =========================================================================

@app.exception_handler(StarletteHTTPException)
async def custom_global_html_error_redirector(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 404 and "text/html" in request.headers.get("accept", ""):
        safe_fallback_payload = {
            "name": "System Security Administrator",
            "config": {"term_name": "Active System Error State", "attendance_threshold": 75.0, "cgpa_threshold": 6.5},
            "security_logs": [], "performance": {"active_connections": 0, "response_time_ms": 0.0},
            "counseling_cases_count": 0, "students": [], "mentors": [], "counselors": [], "total_users": 0
        }
        return templates.TemplateResponse(request=request, name="admin.html", context={"request": request, "data": safe_fallback_payload, "err": f"404 Pathway '{request.url.path}' does not exist.", "current_user": {"role": "admin", "department": "N/A"}}, status_code=404)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

# =========================================================================
#        Current Active Term Progression State Machine Migration (Superadmin Exclusive)
# =========================================================================
from fastapi import APIRouter, status, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
import models
from database import get_db
#from security import get_current_active_admin # Adjust to match your existing import

@app.post("/admin/infrastructure/term-shift")
async def execute_global_term_shift_migration(
    request: Request,
    new_term_title: str = Form(...),
    confirmation_passphrase: str = Form(...),
    db: Session = Depends(get_db)
):
    # 1. Identity Verification Check
    # Resolves directly against local utility dependency injection in main.py
    current_user = await get_current_active_admin(request, db)
    if not current_user or current_user.role != 'superadmin':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access Denied: Only Global System Security Superadmins can trigger infrastructure shifts."
        )

    # 2. Security Passphrase Constraint Validation
    if confirmation_passphrase.strip() != "EXECUTE GLOBAL OVERRIDE SHIFT":
        return RedirectResponse(
            url="/admin/config?action=term_shift&err=Invalid+security+passphrase.+Migration+sequence+aborted.",
            status_code=status.HTTP_303_SEE_OTHER
        )

    try:
        # 3. Execute Single-Query State Shifting Architecture
        # Handles incremental progression up to 7, pushes 8 to NULL, and shifts years concurrently
        db.execute(
            text("""
                UPDATE student_academic_state
                SET 
                    current_semester = CASE 
                        WHEN current_semester >= 8 THEN NULL 
                        ELSE current_semester + 1 
                    END,
                    current_year = CASE 
                        WHEN current_semester >= 8 THEN NULL
                        WHEN current_semester + 1 IN (1, 2) THEN 1
                        WHEN current_semester + 1 IN (3, 4) THEN 2
                        WHEN current_semester + 1 IN (5, 6) THEN 3
                        WHEN current_semester + 1 IN (7, 8) THEN 4
                        ELSE current_year
                    END,
                    last_advanced_academic_year = :term_ref,
                    updated_at = NOW()
                WHERE current_semester IS NOT NULL;
            """),
            {"term_ref": new_term_title}
        )
        
        db.commit()
        return RedirectResponse(
            url="/admin/config?action=term_shift&msg=Global+academic+term+shift+completed+successfully.",
            status_code=status.HTTP_303_SEE_OTHER
        )

    except Exception as e:
        db.rollback()
        return RedirectResponse(
            url=f"/admin/config?action=term_shift&err=Migration+failed:+{str(e)}",
            status_code=status.HTTP_303_SEE_OTHER
        )
    
import io
import csv
from datetime import datetime
from fastapi.responses import StreamingResponse

@app.get("/admin/reports/students")
async def export_student_report(request: Request, db: Session = Depends(get_db)):
    user_token = request.cookies.get("session_token")
    current_admin = db.query(models.DbUser).filter(models.DbUser.email == user_token).first()
    if not current_admin or current_admin.role not in ["admin", "superadmin"]:
        return RedirectResponse(url="/auth/login/admin?err=Unauthorized", status_code=303)

    query = db.query(models.DbUser).filter(models.DbUser.role == "student", models.DbUser.is_active == True)
    if current_admin.role != "superadmin":
        query = query.filter(models.DbUser.department == current_admin.department)
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Student ID/USN", "Full Name", "Department Profile", "Status"])
    for s in query.all():
        writer.writerow([s.usn_or_id, s.display_name, s.department or "Unassigned", "Active"])

    response = StreamingResponse(io.BytesIO(output.getvalue().encode("utf-8")), media_type="text/csv")
    dept_lbl = "Global" if current_admin.role == "superadmin" else f"Dept_{current_admin.department}"
    response.headers["Content-Disposition"] = f"attachment; filename=Student_Academics_{dept_lbl}_{datetime.now().strftime('%Y%m%d')}.csv"
    return response

@app.get("/admin/reports/mentors")
async def export_mentor_report(request: Request, db: Session = Depends(get_db)):
    user_token = request.cookies.get("session_token")
    current_admin = db.query(models.DbUser).filter(models.DbUser.email == user_token).first()
    if not current_admin or current_admin.role not in ["admin", "superadmin"]:
        return RedirectResponse(url="/auth/login/admin?err=Unauthorized", status_code=303)

    query = db.query(models.DbUser).filter(models.DbUser.role == "mentor", models.DbUser.is_active == True)
    if current_admin.role != "superadmin":
        query = query.filter(models.DbUser.department == current_admin.department)
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Mentor Identifier", "Faculty Name", "Department Profile"])
    for m in query.all():
        writer.writerow([m.usn_or_id, m.display_name, m.department or "Unassigned"])

    response = StreamingResponse(io.BytesIO(output.getvalue().encode("utf-8")), media_type="text/csv")
    dept_lbl = "Global" if current_admin.role == "superadmin" else f"Dept_{current_admin.department}"
    response.headers["Content-Disposition"] = f"attachment; filename=Mentor_Report_{dept_lbl}_{datetime.now().strftime('%Y%m%d')}.csv"
    return response

@app.get("/admin/reports/counselors")
async def export_counselor_report(request: Request, db: Session = Depends(get_db)):
    user_token = request.cookies.get("session_token")
    current_admin = db.query(models.DbUser).filter(models.DbUser.email == user_token).first()
    if not current_admin or current_admin.role not in ["admin", "superadmin"]:
        return RedirectResponse(url="/auth/login/admin?err=Unauthorized", status_code=303)

    query = db.query(models.DbUser).filter(models.DbUser.role == "counselor", models.DbUser.is_active == True)
    if current_admin.role != "superadmin":
        query = query.filter(models.DbUser.department == current_admin.department)
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Counselor Tracking Identifier", "Counselor Name", "Branch Assignment"])
    for c in query.all():
        writer.writerow([c.usn_or_id, c.display_name, c.department or "Unassigned"])

    response = StreamingResponse(io.BytesIO(output.getvalue().encode("utf-8")), media_type="text/csv")
    dept_lbl = "Global" if current_admin.role == "superadmin" else f"Dept_{current_admin.department}"
    response.headers["Content-Disposition"] = f"attachment; filename=Counsellor_Report_{dept_lbl}_{datetime.now().strftime('%Y%m%d')}.csv"
    return response

# =========================================================================
#  STUDENT ACADEMIC PROGRESS REPORT LEDGER ENGINE
# =========================================================================
from fastapi import Query
from fastapi.responses import HTMLResponse
from math import ceil

@app.get("/admin/reports/academic-search", response_class=HTMLResponse)
async def search_student_academic_reports(
    request: Request,
    academic_year: str = Query(None),
    department: str = Query(None),
    semester: int = Query(None),
    page: int = Query(1),
    view_student_id: int = Query(None),
    db: Session = Depends(get_db)
):
    # Authentic Session validation
    user_token = request.cookies.get("session_token")
    if not user_token:
        return RedirectResponse(url="/auth/login/admin?err=Session+Expired", status_code=303)
        
    current_admin = db.query(models.DbUser).filter(models.DbUser.email == user_token).first()
    if not current_admin or current_admin.role not in ["admin", "superadmin"]:
        return RedirectResponse(url="/auth/login/admin?err=Unauthorized", status_code=303)

    # Compile drop-down choices variables
    all_departments = [d[0] for d in db.query(models.DbUser.department).distinct().all() if d[0]]
    academic_years = ["2024-2025", "2025-2026", "2026-2027"]
    
    selected_dept = department if current_admin.role == "superadmin" else current_admin.department

    students_list = []
    total_pages = 1
    limit = 25
    offset = (page - 1) * limit
    selected_student_report = None

    # Progressive filter data list evaluation logic
    if academic_year and semester:
        student_query = db.query(models.DbUser).filter(
            models.DbUser.role == "student",
            models.DbUser.is_active == True
        )
        if selected_dept:
            student_query = student_query.filter(models.DbUser.department == selected_dept)
            
        student_query = student_query.join(
            models.DbCourseRegistration, 
            models.DbCourseRegistration.student_id == models.DbUser.usn_or_id
        ).filter(
            models.DbCourseRegistration.academic_year == academic_year,
            models.DbCourseRegistration.semester == semester
        ).distinct()

        total_students = student_query.count()
        total_pages = ceil(total_students / limit) if total_students > 0 else 1
        students_list = student_query.order_by(models.DbUser.usn_or_id).offset(offset).limit(limit).all()
    selected_report = None
# Detailed report panel execution logic safely querying backlogs & course milestones
    if view_student_id:
        target_student = db.query(models.DbUser).filter(models.DbUser.id == view_student_id).first()
        if target_student:
            # Fetch registrations and eagerly load marks ledger rows to minimize DB queries
            enrollment_records = db.query(models.DbCourseRegistration).filter(
                models.DbCourseRegistration.student_id == target_student.usn_or_id,
                models.DbCourseRegistration.semester == semester,
                models.DbCourseRegistration.academic_year == academic_year
            ).all()
            
            # Process and pivot marks ledger into structured course rows
            processed_courses = []
            for reg in enrollment_records:
                # Initialize default fallback states
                marks_map = {"IA-1": "-", "IA-2": "-", "IA-AVG": "-", "SEE": "-"}
                attendance_values = []
                
                # Pivot rows from the marks ledger list
                for ledger in reg.marks_ledger:
                    eval_type = ledger.evaluation_type.value if hasattr(ledger.evaluation_type, 'value') else str(ledger.evaluation_type)
                    if ledger.marks_obtained is not None:
                        marks_map[eval_type] = ledger.marks_obtained
                    if ledger.attendance_percentage is not None:
                        attendance_values.append(ledger.attendance_percentage)
                
                # Determine attendance (use latest milestone attendance or average)
                attendance = f"{attendance_values[-1]:.1f}%" if attendance_values else "N/A"
                
                # Dynamic Result Logic: Determine Pass/Fail status
                # (Modify thresholds according to your university regulations)
                see_marks = marks_map["SEE"]
                ia_avg = marks_map["IA-AVG"]
                
                if see_marks == "-" or ia_avg == "-":
                    result_status = "Incomplete"
                else:
                    try:
                        # Example criteria: Fail if SEE < 35 or Total (IA_AVG + SEE) < 40
                        total_marks = float(ia_avg) + float(see_marks)
                        if float(see_marks) < 35.0 or total_marks < 40.0:
                            result_status = "Fail"
                        else:
                            result_status = "Pass"
                    except ValueError:
                        result_status = "Pending"

                processed_courses.append({
                    "course_code": reg.course.course_code if reg.course else "UNKNOWN",
                    "course_name": reg.course.course_name if reg.course else "Unknown Course",
                    "ia1": marks_map["IA-1"],
                    "ia2": marks_map["IA-2"],
                    "ia_avg": marks_map["IA-AVG"],
                    "see": marks_map["SEE"],
                    "attendance": attendance,
                    "result": result_status
                })

            backlog_query = db.query(models.DbAcademicRecord).filter(
                models.DbAcademicRecord.student_id == str(target_student.id),
                models.DbAcademicRecord.semester == semester,
                models.DbAcademicRecord.academic_year == academic_year
            ).first()

            backlog_count = backlog_query.backlogs_count if backlog_query else 0

            selected_report = {
                "profile": target_student,
                "courses": processed_courses, # Send pre-formatted course details to UI
                "backlogs": backlog_count
            }

    ui_context = {
        "request": request,
        "admin_role": current_admin.role,
        "admin_dept": current_admin.department,
        "departments": all_departments,
        "academic_years": academic_years,
        "students": students_list,
        "current_page": page,
        "total_pages": total_pages,
        "selected_report": selected_report,
        "filters": {
            "academic_year": academic_year,
            "department": department,
            "semester": semester
        }
    }
    return templates.TemplateResponse(
        name="academic_search.html", 
        context=ui_context, 
        request=request
    )

# =========================================================================
#  STUDENT PORTAL COMPONENT & DATABASE CONFIG OPERATIONS
# =========================================================================

@app.get("/dashboard/student", response_class=HTMLResponse)
async def view_student_dashboard(
    request: Request,
    semester: int = None,
    academic_year: str = None,
    view: str = "performance",  # Controls right canvas switcher (performance vs profile)
    db: Session = Depends(get_db)
):
    # Resolve current authenticated student session context
    user_token = request.cookies.get("session_token")
    if not user_token:
        return RedirectResponse(url="/auth/login/student?err=Session+Expired", status_code=303)

    current_student = db.query(models.DbUser).filter(models.DbUser.email == user_token, models.DbUser.role == "student").first()
    if not current_student:
        return RedirectResponse(url="/auth/login/student?err=Unauthorized", status_code=303)

    # 1. Gather Basic Filter Metadata
    max_sem_query = db.query(func.max(models.DbCourseRegistration.semester)).filter(
        models.DbCourseRegistration.student_id == current_student.usn_or_id
    ).scalar()
    max_semester = max_sem_query if max_sem_query else 1
    selected_semester = semester if semester else max_semester

    if not academic_year:
        ay_query = db.query(models.DbCourseRegistration.academic_year).filter(
            models.DbCourseRegistration.student_id == current_student.usn_or_id
        ).order_by(models.DbCourseRegistration.academic_year.desc()).first()
        academic_year = ay_query[0] if ay_query else "2026-2027"

    # 2. Extract Course Registrations and Track Shortages
    registrations = db.query(models.DbCourseRegistration).filter(
        models.DbCourseRegistration.student_id == current_student.usn_or_id,
        models.DbCourseRegistration.semester == max_semester,
        models.DbCourseRegistration.academic_year == academic_year
    ).all()

    processed_courses = []
    has_attendance_shortage = False
    has_academic_shortage = False

    for reg in registrations:
        marks_map = {"IA-1": "-", "IA-2": "-", "IA-AVG": "-", "SEE": "-"}
        attendance_pct = 100.0 

        for ledger in reg.marks_ledger:
            eval_type = ledger.evaluation_type.value if hasattr(ledger.evaluation_type, 'value') else str(ledger.evaluation_type)
            if ledger.marks_obtained is not None:
                marks_map[eval_type] = ledger.marks_obtained
            if ledger.attendance_percentage is not None:
                attendance_pct = float(ledger.attendance_percentage)

        # Flag attendance shortage below 75%
        if attendance_pct < 75.0:
            has_attendance_shortage = True

        # Flag academic shortages on low IA averages (< 20 marks)
        try:
            if marks_map.get("IA-AVG") != "-" and float(marks_map.get("IA-AVG", 0)) < 20.0:
                has_academic_shortage = True
        except ValueError:
            pass

        see_marks = marks_map["SEE"]
        ia_avg = marks_map["IA-AVG"]
        if see_marks == "-" or ia_avg == "-":
            result_status = "Incomplete"
        else:
            try:
                total_marks = float(ia_avg) + float(see_marks)
                if float(see_marks) < 35.0 or total_marks < 40.0:
                    result_status = "Fail"
                    has_academic_shortage = True
                else:
                    result_status = "Pass"
            except ValueError:
                result_status = "Pending"

        processed_courses.append({
            "course_code": reg.course.course_code if reg.course else "UNKNOWN",
            "course_name": reg.course.course_name if reg.course else "Unknown Course",
            "ia1": marks_map["IA-1"],
            "ia2": marks_map["IA-2"],
            "ia_avg": marks_map["IA-AVG"],
            "see": marks_map["SEE"],
            "attendance": f"{attendance_pct}%",
            "attendance_raw": attendance_pct,
            "result": result_status
        })

    # Pull accurate backlog metrics
    backlog_query = db.query(models.DbAcademicRecord).filter(
        models.DbAcademicRecord.student_id == current_student.usn_or_id,
        models.DbAcademicRecord.semester == selected_semester,
        models.DbAcademicRecord.academic_year == academic_year
    ).first()
    backlog_count = backlog_query.backlogs_count if backlog_query else 0
    if backlog_count > 0:
        has_academic_shortage = True

    # BYPASS LAZY-LOAD EXCEPTION: Query explicitly via String match comparison
    profile = db.query(models.DbStudentInfo).filter(
        models.DbStudentInfo.student_id == str(current_student.usn_or_id)
    ).first()

    return templates.TemplateResponse(
        request=request, 
        name="student.html",
        context={
            "student": current_student,
            "profile": profile,
            "courses": processed_courses,
            "backlogs": backlog_count,
            "current_view": view,
            "alerts": {
                "attendance_shortage": has_attendance_shortage,
                "academic_shortage": has_academic_shortage
            },
            "filters": {
                "academic_year": academic_year, 
                "selected_semester": selected_semester,
                "max_semester": max_semester    
            }
        }
    )


@app.post("/dashboard/student/profile/update")
async def update_student_profile_details(
    request: Request,
    father_name: str = Form(None),
    mother_name: str = Form(None),
    guardian_name: str = Form(None),
    phone_number: str = Form(None),
    fathers_phone: str = Form(None),
    mothers_phone: str = Form(None),
    address: str = Form(None),
    tenth_percentage: float = Form(None),
    twelfth_percentage: float = Form(None),
    diploma_percentage: float = Form(None),
    db: Session = Depends(get_db)
):
    user_token = request.cookies.get("session_token")
    if not user_token:
        return RedirectResponse(url="/auth/login/student?err=Session+Expired", status_code=303)

    current_student = db.query(models.DbUser).filter(models.DbUser.email == user_token, models.DbUser.role == "student").first()
    if not current_student:
        return RedirectResponse(url="/auth/login/student?err=Unauthorized", status_code=303)

    try:
        profile = db.query(models.DbStudentInfo).filter(models.DbStudentInfo.student_id == str(current_student.usn_or_id)).first()
        if not profile:
            profile = models.DbStudentInfo(student_id=str(current_student.usn_or_id))
            db.add(profile)

        profile.father_name = father_name.strip() if father_name else None
        profile.mother_name = mother_name.strip() if mother_name else None
        profile.guardian_name = guardian_name.strip() if guardian_name else None
        profile.phone_number = phone_number.strip() if phone_number else None
        profile.fathers_phone = fathers_phone.strip() if fathers_phone else None
        profile.mothers_phone = mothers_phone.strip() if mothers_phone else None
        profile.address = address.strip() if address else None
        profile.tenth_percentage = tenth_percentage
        profile.twelfth_percentage = twelfth_percentage
        profile.diploma_percentage = diploma_percentage

        db.commit()
        return RedirectResponse(url="/dashboard/student?view=profile&msg=Profile+updated+successfully", status_code=303)
    except Exception as e:
        db.rollback()
        return RedirectResponse(url=f"/dashboard/student?view=profile&err=Update+failed:+{str(e)}", status_code=303)
    
# =========================================================================
#  STUDENT INTERACTIVE CHATBOT VIRTUAL CONSOLE ENDPOINT
# =========================================================================
import numpy as np
import re
from fastapi import Request, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer
from database import get_db
import models

# Initialize the model using your local files configuration
ml_model = SentenceTransformer('local_model_files', model_kwargs={"local_files_only": True})

DB_RECORDS_CACHE = []
CACHED_EMBEDDINGS = None

def ensure_semantic_cache(db: Session):
    """Loads all rows and bakes their ML embeddings into memory once at startup."""
    global DB_RECORDS_CACHE, CACHED_EMBEDDINGS
    if not DB_RECORDS_CACHE:
        print("Initializing Local ML Semantic Memory Matrix...")
        DB_RECORDS_CACHE = db.query(models.DbClinicalKnowledgeBase).all()
        if DB_RECORDS_CACHE:
            # Join the fields together to give the model rich context
            corpus = [f"{r.predefined_title} {r.keyword_combinations}" for r in DB_RECORDS_CACHE]
            CACHED_EMBEDDINGS = ml_model.encode(corpus, convert_to_numpy=True)
            print(f"Successfully cached {len(DB_RECORDS_CACHE)} semantic rows in memory.")

@app.post("/dashboard/student/chatbot/query")
async def handle_student_chatbot_query(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    user_message = body.get("message", "").strip()
    
    if not user_message:
        return JSONResponse(content={"title": "Active", "reply": "-> Please describe your symptoms."})

    user_lower = user_message.lower()

    if any(w in user_lower for w in ["breakdown", "collapse", "breaking point", "cannot cope"]):
        return JSONResponse(content={
            "title": "🚨 Severe Overwhelm De-escalation Protocol",
            "reply": "-> Clinical Insight: You are experiencing an acute nervous system overload.\n-> Immediate Somatic Reset: Drop your shoulders. Unclench your jaw. Exhale slowly for a full 6 seconds.",
            "action_url": "https://university.edu/support/mental_health"
        })

    if any(w in user_lower for w in ["exam", "test", "midterm", "finals", "prepared", "study", "failing"]):
        return JSONResponse(content={
            "title": "Academic Evaluation Anxiety Reframe",
            "reply": "-> Clinical Insight: Feeling unprepared triggers an evolutionary threat response that freezes your memory.",
            "action_url": "https://university.edu/support/academic"
        })

    try:
        ensure_semantic_cache(db)
        if DB_RECORDS_CACHE and CACHED_EMBEDDINGS is not None and ml_model is not None:
            query_embedding = ml_model.encode(user_message, convert_to_numpy=True)
            dot_products = np.dot(CACHED_EMBEDDINGS, query_embedding)
            matrix_norms = np.linalg.norm(CACHED_EMBEDDINGS, axis=1)
            query_norm = np.linalg.norm(query_embedding)
            similarity_scores = dot_products / (matrix_norms * query_norm + 1e-8)
            
            best_match_idx = int(np.argmax(similarity_scores))
            highest_score = float(similarity_scores[best_match_idx])
            
            if highest_score > 0.25:
                matched_row = DB_RECORDS_CACHE[best_match_idx]
                return JSONResponse(content={
                    "title": matched_row.predefined_title,
                    "reply": matched_row.predefined_answer,
                    "action_url": matched_row.action_url
                })
    except Exception as e:
        print(f"ML Lookup failed: {e}")

    return JSONResponse(content={
        "title": "Wellness Catalog Matrix",
        "reply": "-> To unlock tailored matching strategies, use clear terms like 'exam stress' or 'sleep issues'.",
        "action_url": "https://university.edu/support"
    })

# =========================================================================
#  SECURE FACULTY COCKPIT ROUTING PORTAL
# =========================================================================

@app.get("/dashboard/mentor", response_class=HTMLResponse)
async def mentor_dashboard_terminal(
    request: Request,
    view: Optional[str] = "cohort",
    selected_student_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Validates token sessions, isolates assigned mentee clusters, and flags
    real-time risks when attendance < 75% or cumulative CGPA < 6.5.
    """
    # 1. Capture and decode active security security session token parameters
    token = request.cookies.get("session_token")
    if not token:
        return RedirectResponse(url="/auth/login/mentor?err=Session+Expired", status_code=303)
        
    current_mentor = db.query(models.DbUser).filter(
        models.DbUser.email == token, 
        models.DbUser.role == "mentor"
    ).first()
    
    if not current_mentor:
        return RedirectResponse(url="/auth/login/mentor?err=Unauthorized+Node+Access", status_code=303)

    # 2. Extract only students mapped explicitly to this supervisor profile
    mappings = db.query(models.MentorStudentMapping).filter(
        models.MentorStudentMapping.mentor_id == current_mentor.usn_or_id,
        models.MentorStudentMapping.is_active == True
    ).all()

    processed_cohort = []
    
    # NEW: Keep a dedicated reference array cleanly driving the selection picker menu
    cohort_dropdown_list = []
    
    for entry in mappings:
        student_user = db.query(models.DbUser).filter(
            models.DbUser.usn_or_id == entry.student_usn_or_id,
            models.DbUser.role == "student"
        ).first()
        
        if not student_user:
            continue

        # Append to picker dropdown list non-destructively
        cohort_dropdown_list.append({
            "usn": student_user.usn_or_id,
            "name": student_user.display_name or student_user.email
        })

        # Fetch latest Semester CGPA entry
        latest_record = db.query(models.DbAcademicRecord).filter(
            models.DbAcademicRecord.student_id == student_user.usn_or_id
        ).order_by(models.DbAcademicRecord.id.desc()).first()
        cgpa_val = float(latest_record.cgpa) if latest_record and latest_record.cgpa else 0.0

        # Calculate Mean Attendance aggregate matrix across courses
        registrations = db.query(models.DbCourseRegistration).filter(
            models.DbCourseRegistration.student_id == student_user.usn_or_id
        ).all()
        
        total_attendance = 0.0
        attendance_count = 0

        for reg in registrations:
            for ledger in reg.marks_ledger:
                if ledger.attendance_percentage is not None:
                    total_attendance += float(ledger.attendance_percentage)
                    attendance_count += 1

        avg_attendance = round(total_attendance / attendance_count, 2) if attendance_count else 0.0

        # 3. Evaluate System Performance Constraint Flags
        is_attendance_low = avg_attendance < 75.0
        is_cgpa_low = cgpa_val < 6.5
        
        if is_attendance_low or is_cgpa_low:
            risk_status = "CRITICAL RISK"
            color_theme = "red"
        else:
            risk_status = "CLEAR / STABLE"
            color_theme = "emerald"

        # 4. Pull dynamic counselor assignment profile fields
        profile_info = student_user.student_profile
        counselor_status = "Not Transferred"
        if profile_info and getattr(profile_info, 'is_transferred_to_counselor', False):
            counselor_status = "Transferred to Counselor"

        processed_cohort.append({
            "usn": student_user.usn_or_id,
            "name": student_user.display_name or student_user.email,
            "department": student_user.department or "General",
            "cgpa": round(cgpa_val, 2),
            "attendance": round(avg_attendance, 1),
            "risk": risk_status,
            "color": color_theme,
            "counselor_status": counselor_status
        })
        
    # FIX: Query mentor_details using current_mentor instead of current_user (placed cleanly out of the loop)
    mentor_details_record = db.query(models.DbMentorDetail).filter(
        models.DbMentorDetail.employee_id == current_mentor.usn_or_id
    ).first() 

# ─── ADVANCED MULTI-SEMESTER TIMELINE CHRONOLOGY TRACKER ───
    semester_records = {}
    selected_student = None
    backlog_count = 0

    if view == "mentee_analytics" and selected_student_id:
        selected_student = db.query(models.DbUser).filter(
            models.DbUser.usn_or_id == selected_student_id,
            models.DbUser.role == "student"
        ).first()

        if selected_student:
            # Query every course registration map row linked to the target student usn
            registrations = db.query(models.DbCourseRegistration).filter(
                models.DbCourseRegistration.student_id == selected_student.usn_or_id
            ).all()

            for reg in registrations:
                # FIX: Fallback safely to global course table definition if registration row is unassigned
                sem = reg.semester
                if (sem is None or sem == 0) and reg.course:
                    sem = reg.course.assigned_semester
                if not sem:
                    sem = 1 # Absolute baseline fallback anchor

                # Convert semester identifier to regular integer for standard sorting matrices
                sem = int(sem)
                if sem not in semester_records:
                    semester_records[sem] = []

                # Initialize variables to extract vertical rows horizontally
                ia1_val = None
                ia2_val = None
                ia_avg = None
                see_val = None
                attendance_val = None

                # Pivot matching score records from the related marks ledger list
                for ledger in reg.marks_ledger:
                    if ledger.attendance_percentage is not None:
                        attendance_val = float(ledger.attendance_percentage)
                    
                    if ledger.evaluation_type == models.EvaluationTypeEnum.IA1:
                        ia1_val = float(ledger.marks_obtained) if ledger.marks_obtained is not None else None
                    elif ledger.evaluation_type == models.EvaluationTypeEnum.IA2:
                        ia2_val = float(ledger.marks_obtained) if ledger.marks_obtained is not None else None
                    elif ledger.evaluation_type == models.EvaluationTypeEnum.IA_AVG:
                        ia_avg = float(ledger.marks_obtained) if ledger.marks_obtained is not None else None
                    elif ledger.evaluation_type == models.EvaluationTypeEnum.SEE:
                        see_val = float(ledger.marks_obtained) if ledger.marks_obtained is not None else None

                # Compute Internal Assessment mid-term mean dynamically if not explicitly saved
                if ia_avg is None and (ia1_val is not None or ia2_val is not None):
                    vals = [v for v in [ia1_val, ia2_val] if v is not None]
                    ia_avg = round(sum(vals) / len(vals), 2)

                # Determine final pass or backlog criteria thresholds
                result_status = "PENDING"
                if see_val is not None and ia_avg is not None:
                    result_status = "PASS" if (ia_avg + see_val) >= 40 else "FAIL"
                
                if result_status == "FAIL":
                    backlog_count += 1

                semester_records[sem].append({
                    "course_code": reg.course_id,
                    "course_name": reg.course.course_name if reg.course else "Academic Course Node",
                    "attendance": attendance_val,
                    "ia1": ia1_val,
                    "ia2": ia2_val,
                    "ia_avg": ia_avg,
                    "see": see_val,
                    "result": result_status
                })
    # ──────────────────────────────────────────────────────────────

    return templates.TemplateResponse(
        request=request,
        name="mentor.html",
        context={
            "mentor": current_mentor,
            "cohort": processed_cohort,
            "cohort_list": cohort_dropdown_list, # <--- Added context variable
            "mentor_details": mentor_details_record, 
            "view": view,
            "selected_student": selected_student, # <--- Added context variable
            "semester_records": semester_records, # <--- Added context variable
            "backlog_count": backlog_count,       # <--- Added context variable
            "data": {
                "cohort_size": len(processed_cohort)
            }
        }
    )
from fastapi import Form

@app.post("/mentor/reports/save")
async def save_mentoring_report(
    student_id: str = Form(...),
    mentor_id: str = Form(...),
    discussion_points: str = Form(...),
    action_items: str = Form(""),
    current_semester: str = Form(...),  # Form field sent from frontend
    db: Session = Depends(get_db)
):
    try:
        # Calculate academic year safely for the filter view
        now = datetime.now()
        current_year = now.year
        current_month = now.month
        if current_month >= 7:  # July or later
            acad_year = f"{current_year} - {current_year + 1}"
        else:
            acad_year = f"{current_year - 1} - {current_year}"

        # Instantiate model instance
        new_report = models.MentoringReport(
            student_id=student_id,
            mentor_id=mentor_id,
            discussion_points=discussion_points,
            action_items=action_items,
            current_semester=int(current_semester) if current_semester.isdigit() else 1, # Safe integer convert
            academic_year=acad_year
        )
        
        db.add(new_report)
        db.commit()
        return {"status": "success", "message": "Report logged successfully!"}
        
    except Exception as e:
        db.rollback()
        print(f"Database Error: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


# Inside main.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import models  # Assuming your models.py file is imported as models

# Add this route to your mentor routes section
@app.get("/mentor/student-semester/{student_id}")
async def get_student_semester(student_id: str, db: Session = Depends(get_db)):
    # Query the student_academic_state table matching the student's USN / id
    academic_state = db.query(models.StudentAcademicState).filter(
        models.StudentAcademicState.student_id == student_id
    ).first()
    
    if not academic_state:
        # Fallback if no specific state record exists yet
        return {"current_semester": "1"}
        
    return {"current_semester": str(academic_state.current_semester)}

from datetime import datetime

@app.get("/mentor/reports/{student_id}")
async def get_mentee_history(
    student_id: str, 
    academic_year: Optional[str] = None, 
    db: Session = Depends(get_db)
):
    query = db.query(models.MentoringReport).filter(
        models.MentoringReport.student_id == student_id
    )
    
    if academic_year:
        # Filter by string match OR fallback to checking timestamp year if the column is empty
        try:
            start_year = int(academic_year.split(" - ")[0])
            # A typical academic cycle runs from July (Month 7) to June of next year
            start_date = datetime(start_year, 7, 1)
            end_date = datetime(start_year + 1, 6, 30, 23, 59, 59)
            
            query = query.filter(
                (models.MentoringReport.academic_year == academic_year) |
                ((models.MentoringReport.academic_year == None) & 
                 (models.MentoringReport.meeting_date >= start_date) & 
                 (models.MentoringReport.meeting_date <= end_date))
            )
        except Exception:
            # If parsing string splits fails, fall back to simple text match criteria
            query = query.filter(models.MentoringReport.academic_year == academic_year)
        
    reports = query.order_by(models.MentoringReport.meeting_date.desc()).all()
    
    output = []
    for r in reports:
        output.append({
            "date": r.meeting_date.strftime("%Y-%m-%d %H:%M") if r.meeting_date else "N/A",
            "points": r.discussion_points,
            "actions": r.action_items or "None assigned"
        })
        
    return output

from fastapi import Form
from fastapi.responses import RedirectResponse

@app.post("/dashboard/mentor/profile/update")
async def update_mentor_profile(
    request: Request,
    name: str = Form(...),
    gender: str = Form(...),
    department_name: str = Form(...),
    designation: str = Form(...),
    highest_qualification: str = Form(""),
    phone_number: str = Form(""),
    pan_card_number: str = Form(""),
    aadhaar_card_number: str = Form(""),
    date_of_joining: datetime = Form(...),
    experience_in_college: int = Form(0),
    teaching_experience_years: float = Form(0.0),
    research_experience_years: float = Form(0.0),
    industry_experience_years: float = Form(0.0),
    other_experience_years: float = Form(0.0),
    db: Session = Depends(get_db)
):
    token = request.cookies.get("session_token")
    if not token:
        return RedirectResponse(url="/auth/login/mentor?err=Session+Expired", status_code=303)
        
    current_mentor = db.query(models.DbUser).filter(
        models.DbUser.email == token, 
        models.DbUser.role == "mentor"
    ).first()
    
    if not current_mentor:
        return RedirectResponse(url="/auth/login/mentor?err=Unauthorized", status_code=303)
        
    # Query your verified DbMentorDetail table instance
    detail_record = db.query(models.DbMentorDetail).filter(
        models.DbMentorDetail.employee_id == current_mentor.usn_or_id
    ).first()
    
    if detail_record:
        # Map values safely to attributes
        detail_record.name = name
        detail_record.gender = gender
        detail_record.department_name = department_name
        detail_record.designation = designation
        detail_record.highest_qualification = highest_qualification
        detail_record.phone_number = phone_number
        detail_record.pan_card_number = pan_card_number
        detail_record.aadhaar_card_number = aadhaar_card_number
        detail_record.date_of_joining = date_of_joining
        detail_record.experience_in_college = experience_in_college
        detail_record.teaching_experience_years = teaching_experience_years
        detail_record.research_experience_years = research_experience_years
        detail_record.industry_experience_years = industry_experience_years
        detail_record.other_experience_years = other_experience_years
        detail_record.total_work_experience_years = round(
            teaching_experience_years + research_experience_years + industry_experience_years + other_experience_years, 2
        )
        
        db.commit()
        # At the end of your update_mentor_profile endpoint, change the return statement to:
    return RedirectResponse(url="/dashboard/mentor?view=profile&success=ProfileUpdated", status_code=303)
