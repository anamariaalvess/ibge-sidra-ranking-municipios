from __future__ import annotations

import re
from typing import Optional, Tuple
import pandas as pd
import numpy as np


# ---------------------------------------------------------
# Helpers: detecção de colunas (SIDRA varia bastante)
# ---------------------------------------------------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def _find_col(df: pd.DataFrame, must_contain: list[str], any_of: Optional[list[str]] = None) -> Optional[str]:
    """
    Procura uma coluna cujo nome contenha TODOS os termos em must_contain
    e (opcionalmente) pelo menos um termo em any_of.
    Retorna o nome da coluna (string) ou None.
    """
    cols = list(df.columns)
    for c in cols:
        cn = _norm(c)
        ok = all(t in cn for t in must_contain)
        if not ok:
            continue
        if any_of is not None and not any(t in cn for t in any_of):
            continue
        return c
    return None


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """
    Converte strings numéricas com vírgula/pontos para float.
    """
    s = series.astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def _drop_header_like_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    O sidrapy às vezes retorna uma primeira linha 'cabeçalho' embutida nos dados.
    Regra simples: manter apenas linhas com V numérico (ou onde existir).
    """
    if "V" in df.columns:
        v = _coerce_numeric(df["V"])
        return df.loc[v.notna()].copy()
    return df.copy()


def _extract_muni_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Tenta achar colunas de município (código e nome).
    """
    # código do município: normalmente contém "munic" e "(código)" ou "codigo"
    muni_code = _find_col(df, must_contain=["munic"], any_of=["código", "codigo"])
    # nome do município: "munic" mas sem "código"
    muni_name = None
    for c in df.columns:
        cn = _norm(c)
        if "munic" in cn and ("código" not in cn and "codigo" not in cn):
            muni_name = c
            break
    return muni_code, muni_name


def _extract_time_col(df: pd.DataFrame) -> Optional[str]:
    """
    Tenta achar uma coluna temporal: Ano, Mês, Trimestre.
    Para este projeto, preferimos Ano.
    """
    # prioridade: ano
    ano = _find_col(df, must_contain=["ano"], any_of=["código", "codigo"]) or _find_col(df, must_contain=["ano"])
    if ano:
        return ano
    # fallback: mês
    mes = _find_col(df, must_contain=["mês"]) or _find_col(df, must_contain=["mes"])
    if mes:
        return mes
    # fallback: período
    per = _find_col(df, must_contain=["período"]) or _find_col(df, must_contain=["periodo"])
    return per


# ---------------------------------------------------------
# Normalização SIDRA -> formato tidy
# ---------------------------------------------------------
def normalize_sidra_table(
    df_raw: pd.DataFrame,
    value_name: str,
    keep_extra_dims: bool = False,
) -> pd.DataFrame:
    """
    Converte o output do sidrapy (tabela SIDRA) para um df "tidy":
    colunas mínimas: code_muni, municipio, ano (se existir), value_name

    Parâmetros:
        - value_name: nome do indicador final (ex.: 'pib', 'pop', 'area')
        - keep_extra_dims: se True, mantém dimensões adicionais (ex.: grupo, variável)
    """
    df = df_raw.copy()
    df = _drop_header_like_rows(df)

    # município
    muni_code_col, muni_name_col = _extract_muni_cols(df)

    # tempo
    time_col = _extract_time_col(df)

    # valor
    if "V" not in df.columns:
        raise ValueError("Não encontrei a coluna 'V' (valor) no retorno do SIDRA.")
    df["__value__"] = _coerce_numeric(df["V"])

    # Monta base
    out = pd.DataFrame()

    if muni_code_col:
        out["code_muni"] = pd.to_numeric(df[muni_code_col], errors="coerce").astype("Int64")
    else:
        # alguns retornos podem vir com "Território (Código)" (menos comum pra municipal)
        terr_code = _find_col(df, must_contain=["territ"], any_of=["código", "codigo"])
        if terr_code:
            out["code_muni"] = pd.to_numeric(df[terr_code], errors="coerce").astype("Int64")
        else:
            out["code_muni"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    if muni_name_col:
        out["municipio"] = df[muni_name_col].astype(str)
    else:
        terr_name = None
        for c in df.columns:
            cn = _norm(c)
            if "territ" in cn and ("código" not in cn and "codigo" not in cn):
                terr_name = c
                break
        out["municipio"] = df[terr_name].astype(str) if terr_name else pd.Series([""] * len(df))

    if time_col:
        # tenta extrair o ano numérico
        out["ano"] = pd.to_numeric(df[time_col], errors="coerce").astype("Int64")
    else:
        out["ano"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    out[value_name] = df["__value__"]

    # dimensões extras (opcional)
    if keep_extra_dims:
        # mantém todas colunas D* e N* comuns no SIDRA
        extra = {}
        for c in df.columns:
            if re.match(r"^D\d+C$", str(c)) or re.match(r"^D\d+N$", str(c)):
                extra[c] = df[c]
        for k, v in extra.items():
            out[k] = v

    # limpa: remove linhas sem valor
    out = out.loc[out[value_name].notna()].copy()

    # garante tipos
    out["code_muni"] = out["code_muni"].astype("Int64")
    out["ano"] = out["ano"].astype("Int64")

    return out


# ---------------------------------------------------------
# Construção do painel final (merge + features derivadas)
# ---------------------------------------------------------
def build_panel(
    mun_uf: pd.DataFrame,
    pib_raw: pd.DataFrame,
    pop_raw: pd.DataFrame,
    area_raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Constrói painel final por município-ano:
        - pib
        - pop
        - area
        - pib_pc (PIB per capita)
        - pib_km2 (PIB por km²)
    """
    mun = mun_uf.copy()
    mun = mun.rename(columns={"id_municipio": "code_muni"}) if "id_municipio" in mun.columns else mun
    mun["code_muni"] = pd.to_numeric(mun["code_muni"], errors="coerce").astype("Int64")

    pib = normalize_sidra_table(pib_raw, value_name="pib")
    pop = normalize_sidra_table(pop_raw, value_name="pop")
    area = normalize_sidra_table(area_raw, value_name="area")

    # AREA normalmente não tem "ano" útil (ou vem só 1). Vamos pegar o maior valor por município.
    area_agg = (
        area.dropna(subset=["code_muni"])
            .groupby("code_muni", as_index=False)["area"]
            .max()
    )

    # merge PIB x POP (por município-ano)
    panel = (
        pib.dropna(subset=["code_muni", "ano"])
            .merge(
                pop.dropna(subset=["code_muni", "ano"]),
                on=["code_muni", "ano"],
                how="inner",
                suffixes=("", "_pop"),
            )
    )

    # adiciona nome do município “oficial” do endpoint de localidades
    panel = panel.merge(mun[["code_muni", "municipio"]], on="code_muni", how="left")

    # adiciona área
    panel = panel.merge(area_agg, on="code_muni", how="left")

    # features derivadas
    panel["pib_pc"] = panel["pib"] / panel["pop"]
    panel["pib_km2"] = np.where(panel["area"].notna() & (panel["area"] > 0), panel["pib"] / panel["area"], np.nan)

    # ordena
    panel = panel.sort_values(["ano", "pib"], ascending=[True, False]).reset_index(drop=True)

    return panel
