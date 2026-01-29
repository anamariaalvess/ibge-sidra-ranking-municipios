from __future__ import annotations
from pathlib import Path
import json
import requests
import pandas as pd
import sidrapy

CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def get_municipios_uf(uf: int) -> pd.DataFrame:
    """
    uf: código da UF (ex.: GO=52).
    Retorna id (código IBGE) e nome do município.
    """
    url = f"https://servicodados.ibge.gov.br/api/v1/localidades/estados/{uf}/municipios"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    return pd.DataFrame([{"id_municipio": int(x["id"]), "municipio": x["nome"]} for x in data])

def sidra_table(table_code: str, territorial_level: str, ibge_codes: str, period: str) -> pd.DataFrame:
    """
    Wrapper simples do sidrapy.get_table.
    """
    raw = sidrapy.get_table(
        table_code=table_code,
        territorial_level=territorial_level,
        ibge_territorial_code=ibge_codes,
        period=period
    )
    return pd.DataFrame(raw)

def fetch_pib_pop_area_go(period_pib="last 10", period_pop="last 10") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # GO = 52 (você pode trocar a UF)
    mun = get_municipios_uf(52)

    # para não estourar URL, você pode pegar só um subconjunto no começo
    # mun = mun.sample(50, random_state=42)

    codes = ",".join(mun["id_municipio"].astype(str).tolist())

    # PIB (tabela 5938), Pop (1301), Área (6579)
    pib  = sidra_table("5938", "6", codes, period_pib)
    pop  = sidra_table("1301", "6", codes, period_pop)
    area = sidra_table("6579", "6", codes, "last 1")  # área não muda tanto

    return mun, pib, pop, area