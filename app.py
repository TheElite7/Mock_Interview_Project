from flask import Flask, request, jsonify, render_template, send_from_directory, redirect
from flask_cors import CORS
import os
import random
import json
import sqlite3
import datetime
from werkzeug.utils import secure_filename
from interview import Interviewer
import requests
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'RECORDINGS')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
DATABASE = 'interview_details.db'

# Initialize database
def initialize_database():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        uname TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        phoneNumber TEXT NOT NULL,
        batch TEXT NOT NULL,
        domain TEXT NOT NULL,
        status TEXT NOT NULL,
        interview_date TEXT,
        completed BOOLEAN DEFAULT FALSE,
        feedback TEXT
    )
    """)
    conn.commit()
    conn.close()

initialize_database()


# Initialize Interviewer
custom_questions = [
    "Tell me about yourself",
    "What are your strengths?",
    "What are your weaknesses?",
    "Why do you want this job?",
    "Where do you see yourself in 5 years?",
    
    "Interview Finished"
]


def load_questions(question_path):
    with open(question_path, 'r') as f:
        questions = [line.strip() for line in f.readlines() if line.strip()]

    first_question = "Tell me about yourself"

    questions = [q for q in questions]

    random.shuffle(questions)
    last_question =  "Interview Finished"
    # return [first_question] + questions
    return[first_question] + questions + [last_question]
def load_feedback(file):
    with open(file , 'r') as f:
        feedback  =  f.read()

    return feedback

custom_questions = load_questions('/home/saaho/Desktop/Mock Interview/Final2/Final/questions.txt')
# feedback_prompt =  load_feedback('/home/saaho/Desktop/Mock Interview/Final2/Mock_Interview_V2/feedbackprompt.txt')
# print(feedback_prompt)
interviewer_instance = Interviewer(job_role='Software Engineer', custom_questions=custom_questions , )

# Routes
@app.route('/')
def details_page():
    return render_template('details.html')

@app.route('/interview')
def interview_page():
    email = request.args.get('email')
    if not email:
        return redirect('/restart')
    
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT completed FROM students WHERE email = ?", (email,))
    result = cursor.fetchone()
    conn.close()
    
    if result and result[0]:  # If interview is completed
        return redirect('/restart')
    
    return render_template('interview.html', email=email)

@app.route('/student_page')
def student_page():
    referrer = request.headers.get("Referer")
    if not referrer or "admin" not in referrer:
        return redirect('/restart')
    return render_template('student.html')

@app.route('/restart')
def restart():
    return render_template('details.html')

@app.route('/add_student', methods=['POST'])
def add_student():
    data = request.json
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO students 
            (uname, email, phoneNumber, batch, domain, status, interview_date, completed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data['uname'],
            data['email'],
            data['phoneNumber'],
            data['batch'],
            data['domain'],
            'pending',
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            False
        ))
        conn.commit()
        return jsonify({
            "status": "success", 
            "message": "Student added.",
            "redirect": f"/interview?email={data['email']}"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    finally:
        conn.close()

@app.route('/start_interview', methods=['POST'])
def start_interview():
    global interviewer_instance
    interviewer_instance.reset()
    interviewer_instance.conduct_interview()
    return jsonify({"status": "success", "message": "Interview started"})

@app.route('/get_questions')
def get_questions():
    return jsonify({
        "current": interviewer_instance.current_question,  
        "history": interviewer_instance.questions_history
    })
@app.route('/stop_interview', methods=['POST'])
def stop_interview():
    email = request.json.get('email')
    if not email:
        return jsonify({"status":"error","message":"Email required"}),400

    interviewer_instance.stop()

    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("UPDATE students SET status=?, completed=1 WHERE email=?",
              ("completed", email))
    # feedback = interviewer_instance.get_feedback_v2()
    # c.execute("UPDATE students SET feedback=? WHERE email=?", (feedback, email))

    conn.commit(); conn.close()

    return jsonify({"status":"success"})


@app.route('/get_feedback', methods=['GET'])
def get_feedback():
  
    feedback=  interviewer_instance.get_feedback_v2()
    return jsonify({"status":"ready","feedback": feedback})



@app.route('/download_feedback', methods=['GET'])
def download_feedback():
    email = request.args.get('email')
    if not email:
        return jsonify({"status": "error", "message": "Email required"}), 400
    
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT uname, domain, feedback FROM students 
        WHERE email = ?
    """, (email,))
    result = cursor.fetchone()
    conn.close()
    
    if not result or not result[2]:
        return jsonify({"status": "error", "message": "Feedback not found"}), 404
    
    name, domain, feedback = result
    feedback_content = f"""
    Interview Feedback Report
    ========================
    
    Candidate: {name}
    Domain: {domain}
    Email: {email}
    Date: {datetime.datetime.now().strftime('%Y-%m-%d')}
    
    {feedback}
    """
    
    return jsonify({
        "status": "success",
        "content": feedback_content,
        "filename": f"{name}_{domain}_feedback.txt"
    })

@app.route('/upload_video', methods=['POST'])
def upload_video():
    file = request.files['video']
    email = request.form.get('email')
    
    if not email:
        return jsonify({"status": "error", "message": "Missing email"}), 400
    
    # Get student details
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT uname, domain FROM students WHERE email = ?", (email,))
    result = cursor.fetchone()
    conn.close()
    
    if not result:
        return jsonify({"status": "error", "message": "Student not found"}), 404
    
    student_name, domain = result
    safe_name = secure_filename(student_name.strip().replace(" ", "_"))
    safe_domain = secure_filename(domain.strip().replace(" ", "_")) if domain else ""
    filename = f"{safe_name}_{safe_domain}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.webm"
    
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)
    
    return jsonify({"status": 'success', "filename": filename})

@app.route('/recordings/<filename>')
def get_video(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)
@app.route('/update_status', methods=['POST'])
def update_status():
    data = request.get_json() or {}
    email = data.get('email')
    if not email:
        return jsonify(status="error", message="Email required"), 400

    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("""
        UPDATE students
        SET completed = 1
        WHERE email = ?
    """, (email,))
    conn.commit()
    conn.close()
    return jsonify(status="success")

if __name__ == "__main__":
    app.run(debug=True)