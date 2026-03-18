# billing_agent.py
import subprocess
import json
import argparse

def run_billing_info_via_ocli(operation: str, personal_account: int) -> dict:
    """Выполняет вычисление, вызывая команду ocli."""
    try:
        # Формируем команду
        cmd = [
            "npx", "openapi-to-cli","billing_info", # Имя сгенерированной команды
            "--type_data", operation,
            "--personal_account", str(personal_account)
        ]

        # Выполняем команду
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True # Вызовет исключение, если ocli вернёт ненулевой код
        )

        # Парсим JSON-ответ от ocli
        response_data = json.loads(result.stdout.strip())
        return response_data

    except subprocess.CalledProcessError as e:
        print(f"Ошибка выполнения ocli: {e.stderr}")
        return {"error": e.stderr}
    except json.JSONDecodeError as e:
        print(f"Ошибка парсинга JSON от ocli: {e}")
        return {"error": f"Invalid JSON response: {result.stdout}"}
    except FileNotFoundError:
        print("Ошибка: команда 'ocli' не найдена. Убедитесь, что ocli установлен и доступен в PATH.")
        return {"error": "ocli command not found"}

if __name__ == "__main__":

    result = run_billing_info_via_ocli("Показания",123456)
    if "error" in result:
        print(f"Произошла ошибка: {result['error']}")
    else:
        print(f"По лицевому счету: {result['personal_account']} (Значение: {result['result']}) (операция: {result['type_request']})")
