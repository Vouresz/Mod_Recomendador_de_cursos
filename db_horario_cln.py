# -*- coding: utf-8 -*-
"""
Script: Procesamiento y convalidación de horarios L3 desde PDF
Entrada:  PDF con horarios (ej. HORARIO 23-2_merged.pdf)
Salida:   horarios_L3_convalidado.csv
Requisitos: pip install pdfplumber pandas openpyxl
"""

import pdfplumber
import pandas as pd
import re

# -----------------------------------------------------------
# 1. TABLA DE EQUIVALENCIAS (Antiguo → Nuevo)
# -----------------------------------------------------------
equivalencias = [
    ("EE250","Dibujo Técnico","CBS01","Fundamentos de Programación"),
    ("EE152","Fund. Ing. Computador","CBS02","Sistemas Operativos I"),
    ("BFI06","Física Moderna","CBN01","Redes de Datos I"),
    ("EE418","Electrónicos I","TLR01","Dispositivos RF"),
    ("EE420","Circuitos II","TLN01","Enrutamiento y Conmutación"),
    ("EE647","Control I","CBS05","Inteligencia Artificial I"),
    ("EE438","Electrónicos II","TLR02","Circuitos RF"),
    ("EE648","Control II","TLN02","Seguridad Redes Empresariales"),
    ("EE467","Protocolos Redes","TLR03","Sistemas de Antenas"),
    ("CIB54","Gestión Redes","TLR04","Redes Inal. Móviles I"),
    ("EE708","Conmutación Telecom","TLR05","Normas Telecom"),
    ("EE445","Fin de Carrera","CIB45","Taller Proyecto Investigación"),
    ("EE446","Proyecto Tesis","CIB46","Taller de Investigación"),
    ("CIB14","Sistemas Operativos","TLN03","Automatización y Programabilidad"),
    ("EE449","Ing. Biomédica","TLN04","Enrutamiento Avanzado"),
    ("EE469","Arquitectura Data Center","TLN05","Servidores y Data Center"),
    ("EE479","Instrumentación Biomédica","TLN06","Monitoreo Redes Empresariales"),
    ("EE548","Redes Inalámbricas","TLR06","Redes Inalámbricas II"),
    ("EE594","Antenas","TLR07","Radioelectrónica Espacial"),
    ("EE663","Compiladores","TLN07","Computación en la Nube"),
    ("EE644","Diseño Lógico","CBG01","Fundamentos Ciberseguridad"),
    ("EE672","Robótica Industrial","TLN08","Redes Virtuales"),
    ("EE677","Robótica Médica","CBS37","Internet de las Cosas"),
    ("EE678","Control de Procesos","CBN03","Seguridad Redes Industriales I"),
    ("EE681","Arquitectura Paralela","CBN06","Seguridad Redes Industriales II"),
    ("EE689","Microcontroladores","CBN07","Seguridad Redes Móviles"),
    ("EE693","Microelectrónica","TLN09","Automatización en la Nube"),
    ("EE718","Control Avanzado","CBS03","Sistemas Operativos II"),
    ("EE735","Control Predictivo","TLR08","Comunicaciones Ópticas"),
    ("EE742","Control Robots Móviles","CBS33","Radio Definida por Software"),
    ("IT134","Microprocesadores","CBS36","Desarrollo Aplicaciones Web")
]

df_equiv = pd.DataFrame(equivalencias,
    columns=["cod_ant","nom_ant","cod_nuevo","nom_nuevo"]
)

# -----------------------------------------------------------
# 2. MALLA COMPLETA L3 (Códigos)
# -----------------------------------------------------------
malla_L3 = [
    # 1ro a 10mo ciclo + Electivos
    "BAE01","BFI01","BIC01","BMA01","BMA03","BRN01","CBS01",
    "BFI05","BMA02","BMA09","BQU01","BRC01","CBS02",
    "BEG01","BFI03","BMA05","BMA10","BMA15","EE306",
    "BEF01","CBN01","BMA07","BMA18","EE320","EE410","BIE01",
    "BMA22","TLR01","TLN01","EE428","EE522","CBS05",
    "EE430","TLR02","EE458","EE588","EE604","TLN02",
    "TLR03","EE530","EE590",
    "BEG06","EE498","EE592",
    "TLR04","CIB45","TLR05",
    "EE712","CIB46",
    # Electivos:
    "BMA20","BMA25","BMT01","BRN35","CI105","CI106","CIB02","CIB06",
    "CIB08","CIB12","TLN03","CIB18","CIB28","CIB32","CIB38","CIB50",
    "CIB59","EC119","TLN04","EE468","TLN05","TLN06","EE508","EE524",
    "TLR06","EE586","EE593","TLR07","EE596","EE598","EE608","TLN07"
]

# -----------------------------------------------------------
# 3. FUNCIONES DE PROCESAMIENTO
# -----------------------------------------------------------
def extraer_tablas_pdf(pdf_file):
    """Extrae tablas del PDF usando pdfplumber."""
    filas = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            tablas = page.extract_tables()
            for tabla in tablas:
                for fila in tabla:
                    filas.append(fila)
    return filas


def limpiar_filas_brutas(filas):
    """Convierte texto bruto del PDF a DataFrame estructurado."""
    registros = []

    # patrones útiles
    codigo_pat = re.compile(r"\b[A-Z]{1,4}\d{2,3}\b")
    hora_pat = re.compile(r"(\d{1,2}:\d{2})(?:\s*-\s*(\d{1,2}:\d{2}))?")

    for f in filas:
        # normalizar a lista de strings y eliminar vacíos
        clean = [str(x).strip() for x in f if x is not None and str(x).strip() != ""]
        if not clean:
            continue

        # buscar índice del código (primera ocurrencia que parezca código)
        idx_codigo = None
        for i, v in enumerate(clean):
            if codigo_pat.search(v):
                idx_codigo = i
                break

        # Si no encontramos código, intentar usar el segundo elemento (fallback)
        if idx_codigo is None and len(clean) > 1:
            idx_codigo = 1
        elif idx_codigo is None:
            idx_codigo = 0

        # asignaciones por posición relativa al código
        sigla = clean[0] if len(clean) > 0 and idx_codigo != 0 else ""
        codigo = clean[idx_codigo] if idx_codigo < len(clean) else ""

        # sección suele estar justo después del código
        seccion = clean[idx_codigo + 1] if idx_codigo + 1 < len(clean) else ""

        # buscar horas (índices)
        idx_hora = None
        for i, v in enumerate(clean):
            if hora_pat.search(v):
                idx_hora = i
                break

        # construir nombre del curso: desde después de la sección hasta antes de la hora o docente
        start_name = idx_codigo + 2
        end_name = idx_hora if idx_hora is not None else min(len(clean), start_name + 2)
        curso_nombre = ""
        if start_name < len(clean):
            curso_nombre = " ".join(clean[start_name:end_name]).strip()

        # docente: intentar el elemento antes de tipo/aula/horas
        posible_doc_idx = end_name
        docente = ""
        if posible_doc_idx < len(clean):
            # si el candidato contiene coma (Formato APELLIDO, NOMBRE) o varias palabras, tomarlo
            cand = clean[posible_doc_idx]
            if "," in cand or (len(cand.split()) >= 2 and not codigo_pat.search(cand)):
                docente = cand
                posible_doc_idx += 1

        # resto: tipo, aula, ciclo, dia
        tipo = clean[posible_doc_idx] if posible_doc_idx < len(clean) else ""
        aula = clean[posible_doc_idx + 1] if posible_doc_idx + 1 < len(clean) else ""
        ciclo = clean[posible_doc_idx + 2] if posible_doc_idx + 2 < len(clean) else ""
        dia = clean[posible_doc_idx + 3] if posible_doc_idx + 3 < len(clean) else ""

        # horas
        hora_ini = ""
        hora_fin = ""
        if idx_hora is not None:
            m = hora_pat.search(clean[idx_hora])
            if m:
                hora_ini = m.group(1)
                hora_fin = m.group(2) if m.group(2) else ""

        registros.append({
            "sigla": sigla,
            "codigo_original": codigo,
            "seccion": seccion,
            "curso_original": curso_nombre,
            "docente": docente,
            "tipo": tipo,
            "aula": aula,
            "ciclo": ciclo,
            "dia": dia,
            "hora_inicio": hora_ini,
            "hora_fin": hora_fin
        })

    return pd.DataFrame(registros)


def aplicar_convalidaciones(df):
    df = df.copy()
    df["codigo_nuevo"] = df["codigo_original"]
    df["curso_nuevo"] = df["curso_original"]

    for _, row in df_equiv.iterrows():
        mask = df["codigo_original"] == row["cod_ant"]
        df.loc[mask, "codigo_nuevo"] = row["cod_nuevo"]
        df.loc[mask, "curso_nuevo"] = row["nom_nuevo"]

    return df


# -----------------------------------------------------------
# 4. PROCESAR PDF
# -----------------------------------------------------------
PDF_INPUT = "HORARIO 23-2.pdf"

print("Extrayendo tablas del PDF...")
filas = extraer_tablas_pdf(PDF_INPUT)

print("Limpiando tabla...")
df = limpiar_filas_brutas(filas)

print("Aplicando convalidaciones...")
df = aplicar_convalidaciones(df)

print("Filtrando solo cursos L3...")
df_L3 = df[df["codigo_nuevo"].isin(malla_L3)].copy()

# -----------------------------------------------------------
# 5. EXPORTAR CSV FINAL
# -----------------------------------------------------------
OUTPUT_CSV = "horarios_L3_convalidado.csv"
# Asegurar columnas en orden esperado (evita columnas vacías desalineadas)
expected_cols = [
    "sigla","codigo_original","seccion","curso_original","docente",
    "tipo","aula","ciclo","dia","hora_inicio","hora_fin",
    "codigo_nuevo","curso_nuevo"
]

# crear columnas faltantes si no existen
for c in expected_cols:
    if c not in df_L3.columns:
        df_L3[c] = ""

df_L3 = df_L3[expected_cols]
df_L3.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

print(f"\n Proceso completado.")
print(f"Archivo CSV generado: {OUTPUT_CSV}")
print(f"Total registros L3: {len(df_L3)}")
