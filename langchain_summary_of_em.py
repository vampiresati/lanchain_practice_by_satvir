from openpyxl import load_workbook


counting_file="last_counting_file.txt"
file_path = "dummy_employees.xlsx"
from langchain.agents import create_agent
def new_employee_summary(employees) -> str:
    agent = create_agent(model="ollama:qwen3:4b")
    system_prompt="You are a helpful assistant to summarize the newly added employees."
    result = agent.invoke({"messages": [{"role": "system", "content": system_prompt},
        {"role": "user", "content": str(employees)}]})
    print(result ) # Answer(summary=..., confidence=...)
    final_summary=result['messages'][-1].content
    return final_summary
def read_employees(file_path: str) -> list[dict]:
    """
    Read employee data from an Excel file.

    Expected columns:
    Employee Name, Employee ID, Employee Role
    """
    workbook = load_workbook(file_path)
    sheet = workbook.active

    rows = list(sheet.iter_rows(values_only=True))

    if not rows:
        return []

    headers = rows[0]

    employees = [
        dict(zip(headers, row))
        for row in rows[1:]
        if any(value is not None for value in row)
    ]
    with open(counting_file, "r") as file:
        file_count = int(file.read())
        print(file_count)
    new_employees=employees[file_count:]
    return employees,new_employees


def add_employee(
    file_path: str,
    employee_name: str,
    employee_id: str,
    employee_role: str,
) -> None:
    """
    Add a new employee to the Excel file.
    """
    workbook = load_workbook(file_path)
    sheet = workbook.active

    sheet.append([
        employee_id,
        employee_name,
        employee_role,
    ])

    workbook.save(file_path)


def update_employee_role(
    file_path: str,
    employee_id: str,
    new_role: str,
) -> bool:
    """
    Update an employee's role using the employee ID.

    Returns True if the employee was found, otherwise False.
    """
    workbook = load_workbook(file_path)
    sheet = workbook.active

    for row in sheet.iter_rows(min_row=2):
        current_employee_id = row[1].value

        if current_employee_id == employee_id:
            row[2].value = new_role
            workbook.save(file_path)
            return True

    return False


# Read data
employees,new_employees = read_employees(file_path)
print(len(employees), "employees found.")
print(type(employees))
for employee in new_employees:
    print(employee)
c=new_employee_summary(new_employees)
print(c)

#
# with open(counting_file, "w") as file:
#     file.write(str(len(employees)))
# # Add data
# add_employee(
#     file_path=file_path,
#     employee_name="st Verma",
#     employee_id="EMP007",
#     employee_role="Data Analyst",
# )

# Update data
# updated = update_employee_role(
#     file_path=file_path,
#     employee_id="EMP002",
#     new_role="Senior HR Manager",
# )
#
# if updated:
#     print("Employee updated successfully.")
# else:
#     print("Employee not found.")
