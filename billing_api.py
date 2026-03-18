# billing_api.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal

# Определяем структуру тела запроса
class BillingRequest(BaseModel):
    type_data: Literal["Начисления", "Показания"] # Разрешаем только начисления или показания
    personal_account: int
    #b: float

# Определяем структуру ответа
class BillingResponse(BaseModel):
    result: float
    type_request: str
    personal_account: int

# Создаём приложение FastAPI
app = FastAPI(
    title="Billing API",
    description="Billing Information API for demonstration.",
    version="1.0.0"
)

# Определяем эндпоинт
@app.post("/billing_info", response_model=BillingResponse)
def Billing_info(request: BillingRequest):
    """
    Performs a basic arithmetic operation.
    """
    if request.type_data == "Начисления":
        result = 10000.23
    elif request.type_data == "Показания":
        result = 55423
        # В реальном приложении нужно обработать ошибку
    else:
        result = 0.0

    return BillingResponse(
        result=result,
        type_request=request.type_data,
        personal_account=request.personal_account
    )

# Точка входа для запуска сервера
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)