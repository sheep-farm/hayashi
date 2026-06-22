"""Gera sample.xlsx para testes do Hayashi. Requer: pip install openpyxl"""
from openpyxl import Workbook

wb = Workbook()
ws = wb.active
ws.title = "Dados"
ws.append(["id", "nome", "valor", "ativo"])
ws.append([1, "Alpha", 100.5, True])
ws.append([2, "Beta",  200.3, False])
ws.append([3, "Gamma", 150.0, True])
ws.append([4, "Delta", 80.7,  False])
ws.append([5, "Epsilon", 300.1, True])

ws2 = wb.create_sheet("Resumo")
ws2.append(["metric", "value"])
ws2.append(["mean", 166.32])
ws2.append(["std",  82.45])
ws2.append(["n", 5])

wb.save("exemplos/data/sample.xlsx")
print("sample.xlsx criado")
