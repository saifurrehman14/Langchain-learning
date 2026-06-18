from langchain_text_splitters import RecursiveCharacterTextSplitter , Language

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size = 300,
    chunk_overlap = 0
)

text = """
import random
import datetime

class Student:
    def __init__(self, name, marks):
        self.name = name
        self.marks = marks

    def average(self):
        return sum(self.marks) / len(self.marks)

    def grade(self):
        avg = self.average()
        if avg >= 80:
            return 'A'
        elif avg >= 60:
            return 'B'
        elif avg >= 50:
            return 'C'
        else:
            return 'F'

def generate_students(n):
    students = []
    for i in range(n):
        name = f"Student_{i+1}"
        marks = [random.randint(40, 100) for _ in range(5)]
        students.append(Student(name, marks))
    return students

def save_results(students, filename):
    with open(filename, 'w') as f:
        for s in students:
            line = f"{s.name},{s.marks},{s.average():.2f},{s.grade()}\n"
            f.write(line)

def display_topper(students):
    topper = max(students, key=lambda s: s.average())
    print("Topper:", topper.name)
    print("Average:", topper.average())
    print("Grade:", topper.grade())

def main():
    print("Generating student data...")
    students = generate_students(10)

    for s in students:
        print(s.name, s.marks, "Avg:", round(s.average(), 2), "Grade:", s.grade())

    filename = "results.txt"
    save_results(students, filename)
    print("Results saved to", filename)

    display_topper(students)

    print("Execution time:", datetime.datetime.now())

if __name__ == "__main__":
    main()
"""

chunk = splitter.split_text(text)
print(chunk[0])
print("---------")
print(chunk[1])
print("---------")
print(chunk[2])
print("---------")
print(chunk[3])
print("---------")
print(chunk[4])
print("---------")
print(chunk[5])
print("---------")
print(chunk[6])
print("---------")