import pandas as pd
import networkx as nx
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple

class CourseRecommendationKG:
    def __init__(self, csv_path: str):
       
        self.df = pd.read_csv(csv_path)
        self.graph = nx.DiGraph()
        self.student_performance = {}
        self._build_knowledge_graph()
        
    def _build_knowledge_graph(self):
        
        # Agregar nodos para cada curso
        for _, row in self.df.iterrows():
            self.graph.add_node(
                row['course_id'],
                name=row['course_name'],
                linea_carrera=row['linea_carrera'],
                ciclo=row['ciclo']
            )
            
            # Agregar aristas de prerrequisitos
            if pd.notna(row['prerequisitos']) and row['prerequisitos'] != '':
                prereqs = str(row['prerequisitos']).split(';')
                for prereq in prereqs:
                    prereq = prereq.strip()
                    if prereq:
                        self.graph.add_edge(
                            prereq, 
                            row['course_id'],
                            relationship='prerequisito'
                        )
    
    def load_student_data(self, student_courses: pd.DataFrame):
        
        self.student_performance = {}
        
        for _, row in student_courses.iterrows():
            course_id = row['course_id']
            nota = row['nota']
            
            # Obtener información del curso
            course_info = self.df[self.df['course_id'] == course_id]
            if not course_info.empty:
                linea = course_info.iloc[0]['linea_carrera']
                
                if linea not in self.student_performance:
                    self.student_performance[linea] = []
                
                self.student_performance[linea].append({
                    'course_id': course_id,
                    'nota': nota
                })
    
    def calculate_performance_by_career_line(self) -> Dict[str, float]:
        
        performance_avg = {}
        
        for linea, courses in self.student_performance.items():
            notas = [c['nota'] for c in courses]
            performance_avg[linea] = np.mean(notas)
        
        return performance_avg
    
    def get_available_courses(self) -> List[str]:
        
        completed_courses = set()
        for courses in self.student_performance.values():
            for course in courses:
                completed_courses.add(course['course_id'])
        
        available = []
        
        for node in self.graph.nodes():
            # Si ya lo completó, no lo recomienda
            if node in completed_courses:
                continue
            
            # Verificar prerrequisitos
            prerequisites = list(self.graph.predecessors(node))
            
            if all(prereq in completed_courses for prereq in prerequisites):
                available.append(node)
        
        return available
    
    def recommend_courses(self, top_n: int = 5) -> List[Dict]:
        
        available_courses = self.get_available_courses()
        performance_by_line = self.calculate_performance_by_career_line()
        
        recommendations = []
        
        for course_id in available_courses:
            course_data = self.df[self.df['course_id'] == course_id].iloc[0]
            linea = course_data['linea_carrera']
            
            # Score base: desempeño en la línea de carrera
            if linea in performance_by_line:
                performance_score = performance_by_line[linea] / 20.0  # Normalizar
            else:
                performance_score = 0.5  # Score neutral si no hay datos
            
            # Factor de importancia: número de cursos que dependen
            successors = list(self.graph.successors(course_id))
            importance_score = len(successors) / 10.0  # Normalizar
            
            # Factor de progresión: ciclo del curso
            ciclo_score = 1.0 - (course_data['ciclo'] / 10.0)  # Priorizar cursos del ciclo actual
            
            # Score final ponderado
            final_score = (
                performance_score * 0.6 +  # 60% basado en desempeño
                importance_score * 0.2 +   # 20% basado en importancia
                ciclo_score * 0.2           # 20% basado en progresión
            )
            
            recommendations.append({
                'course_id': course_id,
                'course_name': course_data['course_name'],
                'linea_carrera': linea,
                'ciclo': course_data['ciclo'],
                'score': final_score,
                'performance_in_line': performance_by_line.get(linea, 'N/A'),
                'reason': self._generate_reason(course_data, performance_by_line.get(linea))
            })
        
        # Ordenar por score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return recommendations[:top_n]
    
    def _generate_reason(self, course_data, avg_performance) -> str:
        
        linea = course_data['linea_carrera']
        
        if avg_performance is None:
            return f"Nuevo en la línea de {linea}. Buen momento para explorar."
        elif avg_performance >= 15:
            return f"Excelente desempeño en {linea} (promedio: {avg_performance:.1f}). Muy recomendado."
        elif avg_performance >= 13:
            return f"Buen desempeño en {linea} (promedio: {avg_performance:.1f}). Recomendado."
        elif avg_performance >= 11:
            return f"Desempeño moderado en {linea} (promedio: {avg_performance:.1f}). Considera reforzar."
        else:
            return f"Desempeño bajo en {linea} (promedio: {avg_performance:.1f}). Requiere refuerzo."
    
    
    def print_recommendations(self, recommendations: List[Dict]):
        print("\n" + "="*80)
        print("RECOMENDACIONES DE CURSOS")
        print("="*80 + "\n")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['course_name']} ({rec['course_id']})")
            print(f"   Línea de Carrera: {rec['linea_carrera']}")
            print(f"   Ciclo: {rec['ciclo']}")
            print(f"   Score: {rec['score']:.3f}")
            print(f"   Razón: {rec['reason']}")
            print()

if __name__ == "__main__":
    # Crear el sistema
    system = CourseRecommendationKG('cursos.csv')
    
    # Ejemplo: datos del estudiante (cursos completados con notas)
    student_data = pd.DataFrame({
        'course_id': ['CS101', 'MATH101', 'CS102'],
        'nota': [16, 14, 15]
    })
    
    # Cargar datos del estudiante
    system.load_student_data(student_data)
    
    # Obtener recomendaciones
    recommendations = system.recommend_courses(top_n=5)
    
    # Mostrar resultados
    system.print_recommendations(recommendations)
    
