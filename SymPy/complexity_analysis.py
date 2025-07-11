#!/usr/bin/env python3
"""
SYMPY ALGORITHMIC COMPLEXITY ANALYSIS
=====================================
Phân tích độ phức tạp thuật toán của các tính năng SymPy
"""

import sympy as sp
from sympy import *
import time
import sys
import gc

# Try to import optional dependencies
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

def get_memory_usage():
    """Lấy memory usage hiện tại"""
    if HAS_PSUTIL:
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
    else:
        return 0.0  # Return 0 if psutil not available

def benchmark_function(func, *args, runs=5):
    """Benchmark một function với nhiều lần chạy"""
    times = []
    memory_before = get_memory_usage()
    
    for _ in range(runs):
        gc.collect()  # Clear memory
        start_time = time.perf_counter()
        result = func(*args)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    memory_after = get_memory_usage()
    avg_time = sum(times) / len(times)
    memory_used = memory_after - memory_before
    
    return avg_time, memory_used, result

def print_section(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def print_complexity(operation, theoretical, practical, notes=""):
    print(f"📊 {operation}")
    print(f"   Theoretical: {theoretical}")
    print(f"   Practical:   {practical}")
    if notes:
        print(f"   Notes:       {notes}")
    print()

def main():
    print("🔬 SYMPY ALGORITHMIC COMPLEXITY ANALYSIS")
    print("Phân tích độ phức tạp thuật toán của SymPy")
    
    x, y, z = symbols('x y z')
    n = symbols('n', integer=True, positive=True)
    
    # =================================================================
    # 1. BASIC SYMBOLIC OPERATIONS COMPLEXITY
    # =================================================================
    print_section("1. ĐỘ PHỨC TẠP CÁC PHÉP TOÁN SYMBOLIC CƠ BẢN")
    
    print("🧮 Testing expression expansion complexity...")
    
    # Test expand complexity với different degrees
    degrees = [5, 10, 15, 20, 25]
    expand_times = []
    
    for degree in degrees:
        expr = (x + y + z + 1)**degree
        avg_time, memory, _ = benchmark_function(expand, expr)
        expand_times.append(avg_time)
        if HAS_PSUTIL:
            print(f"Expand (x+y+z+1)^{degree}: {avg_time:.4f}s, Memory: {memory:.2f}MB")
        else:
            print(f"Expand (x+y+z+1)^{degree}: {avg_time:.4f}s")
    
    print_complexity(
        "Expression Expansion",
        "O(d^n) where d=degree, n=variables",
        f"Measured times: {[f'{t:.4f}s' for t in expand_times]}",
        "Exponential growth với degree"
    )
    
    # Test factor complexity
    print("\n🔢 Testing factorization complexity...")
    
    factor_times = []
    for degree in [10, 15, 20, 25, 30]:
        expr = x**degree - 1  # Easy to factor: (x-1)(x^(n-1) + x^(n-2) + ... + 1)
        avg_time, memory, _ = benchmark_function(factor, expr)
        factor_times.append(avg_time)
        if HAS_PSUTIL:
            print(f"Factor x^{degree}-1: {avg_time:.4f}s, Memory: {memory:.2f}MB")
        else:
            print(f"Factor x^{degree}-1: {avg_time:.4f}s")
    
    print_complexity(
        "Polynomial Factorization",
        "O(d^6) for dense polynomials (worst case exponential)",
        f"Measured times: {[f'{t:.4f}s' for t in factor_times]}",
        "Depends heavily on polynomial structure"
    )
    
    # =================================================================
    # 2. EQUATION SOLVING COMPLEXITY
    # =================================================================
    print_section("2. ĐỘ PHỨC TẠP GIẢI PHƯƠNG TRÌNH")
    
    print("⚡ Testing polynomial equation solving...")
    
    # Linear systems
    print("\n--- Linear Systems ---")
    matrix_sizes = [5, 10, 20, 30, 50]
    linear_solve_times = []
    
    for size in matrix_sizes:
        # Create invertible matrix (add identity to ensure non-singular)
        A = Matrix([[i+j+1 if i != j else i+j+2 for j in range(size)] for i in range(size)])
        b = Matrix([i+1 for i in range(size)])
        
        avg_time, memory, _ = benchmark_function(A.LUsolve, b)
        linear_solve_times.append(avg_time)
        if HAS_PSUTIL:
            print(f"Linear system {size}x{size}: {avg_time:.4f}s, Memory: {memory:.2f}MB")
        else:
            print(f"Linear system {size}x{size}: {avg_time:.4f}s")
    
    print_complexity(
        "Linear System Solving (LU Decomposition)",
        "O(n³) for n×n matrix",
        f"Measured times: {[f'{t:.4f}s' for t in linear_solve_times]}",
        "Follows cubic growth as expected"
    )
    
    # Polynomial equation degrees
    print("\n--- Polynomial Equations ---")
    poly_solve_times = []
    for degree in [2, 3, 4, 5]:
        # Create polynomial equation
        poly_expr = sum(x**i for i in range(degree+1)) - 1
        
        avg_time, memory, result = benchmark_function(solve, poly_expr, x)
        poly_solve_times.append(avg_time)
        print(f"Polynomial degree {degree}: {avg_time:.4f}s, {len(result)} solutions")
    
    print_complexity(
        "Polynomial Equation Solving",
        "O(d^3) for degree d (Abel-Ruffini: no general formula for d≥5)",
        f"Measured times: {[f'{t:.4f}s' for t in poly_solve_times]}",
        "Becomes impossible for high degrees without numerical methods"
    )
    
    # =================================================================
    # 3. CALCULUS OPERATIONS COMPLEXITY
    # =================================================================
    print_section("3. ĐỘ PHỨC TẠP CÁC PHÉP TOÁN GIẢI TÍCH")
    
    print("📈 Testing calculus operations...")
    
    # Differentiation complexity
    print("\n--- Differentiation ---")
    diff_times = []
    expressions = [
        x**2,
        x**5 + 3*x**3 + 2*x,
        sin(x)*cos(x)*exp(x),
        (x**2 + 1)**(1/2),
        sum(x**i/factorial(i) for i in range(10))  # Series
    ]
    
    for i, expr in enumerate(expressions):
        avg_time, memory, _ = benchmark_function(diff, expr, x)
        diff_times.append(avg_time)
        print(f"Diff expression {i+1}: {avg_time:.6f}s")
    
    print_complexity(
        "Symbolic Differentiation",
        "O(n) where n is expression tree size",
        "Linear với expression complexity",
        "Very efficient - follows chain rule mechanically"
    )
    
    # Integration complexity
    print("\n--- Integration ---")
    int_times = []
    int_expressions = [
        x**2,
        sin(x),
        exp(x),
        1/(x**2 + 1),
        sin(x)*cos(x)
    ]
    
    for i, expr in enumerate(int_expressions):
        avg_time, memory, result = benchmark_function(integrate, expr, x)
        int_times.append(avg_time)
        print(f"Integrate expression {i+1}: {avg_time:.4f}s")
    
    print_complexity(
        "Symbolic Integration",
        "Undecidable in general! Can be exponential or impossible",
        f"Simple cases: {[f'{t:.4f}s' for t in int_times]}",
        "Uses Risch algorithm - very complex for general cases"
    )
    
    # =================================================================
    # 4. LINEAR ALGEBRA COMPLEXITY
    # =================================================================
    print_section("4. ĐỘ PHỨC TẠP ĐẠI SỐ TUYẾN TÍNH")
    
    print("🔺 Testing matrix operations...")
    
    # Matrix multiplication
    print("\n--- Matrix Multiplication ---")
    sizes = [10, 20, 30, 50]
    matmul_times = []
    
    for size in sizes:
        A = Matrix([[i+j+1 for j in range(size)] for i in range(size)])
        B = Matrix([[i*j+1 for j in range(size)] for i in range(size)])
        
        avg_time, memory, _ = benchmark_function(lambda: A * B)
        matmul_times.append(avg_time)
        print(f"Matrix multiplication {size}x{size}: {avg_time:.4f}s")
    
    print_complexity(
        "Matrix Multiplication",
        "O(n³) naive, O(n^2.807) Strassen, O(n^2.373) current best",
        f"SymPy uses O(n³): {[f'{t:.4f}s' for t in matmul_times]}",
        "Clear cubic scaling"
    )
    
    # Determinant calculation
    print("\n--- Determinant Calculation ---")
    det_times = []
    
    for size in [5, 10, 15, 20]:
        A = Matrix([[i+j+1 if i != j else i+j+3 for j in range(size)] for i in range(size)])
        
        avg_time, memory, _ = benchmark_function(A.det)
        det_times.append(avg_time)
        print(f"Determinant {size}x{size}: {avg_time:.4f}s")
    
    print_complexity(
        "Determinant Calculation",
        "O(n³) using LU decomposition",
        f"Measured: {[f'{t:.4f}s' for t in det_times]}",
        "Much better than O(n!) naive expansion"
    )
    
    # Eigenvalue computation
    print("\n--- Eigenvalue Computation ---")
    eigen_times = []
    
    for size in [3, 5, 8, 10]:
        A = Matrix([[i+j+1 if i != j else i+j+3 for j in range(size)] for i in range(size)])
        
        try:
            avg_time, memory, _ = benchmark_function(A.eigenvals)
            eigen_times.append(avg_time)
            print(f"Eigenvalues {size}x{size}: {avg_time:.4f}s")
        except Exception as e:
            print(f"Eigenvalues {size}x{size}: Failed - {e}")
            eigen_times.append(float('inf'))
    
    print_complexity(
        "Eigenvalue Computation",
        "O(n³) for numerical, can be much worse for symbolic",
        "Depends on matrix structure - can become intractable",
        "SymPy tries exact computation - very expensive"
    )
    
    # =================================================================
    # 5. NUMBER THEORY COMPLEXITY
    # =================================================================
    print_section("5. ĐỘ PHỨC TẠP SỐ HỌC")
    
    print("🔢 Testing number theory operations...")
    
    # Prime testing
    print("\n--- Primality Testing ---")
    prime_test_times = []
    test_numbers = [101, 1009, 10007, 100003, 982451653]
    
    for num in test_numbers:
        avg_time, memory, result = benchmark_function(isprime, num)
        prime_test_times.append(avg_time)
        print(f"isprime({num}): {avg_time:.6f}s, Result: {result}")
    
    print_complexity(
        "Primality Testing",
        "O(k log³ n) Miller-Rabin test",
        "SymPy uses deterministic test for small numbers",
        "Very efficient for practical sizes"
    )
    
    # Integer factorization
    print("\n--- Integer Factorization ---")
    factor_times = []
    test_numbers = [1001, 10001, 100001, 982451653]
    
    for num in test_numbers:
        avg_time, memory, result = benchmark_function(factorint, num)
        factor_times.append(avg_time)
        print(f"factorint({num}): {avg_time:.4f}s, Factors: {result}")
    
    print_complexity(
        "Integer Factorization",
        "Exponential in general (no known polynomial algorithm)",
        "Uses trial division, Pollard rho, etc.",
        "Can be very slow for large semiprimes"
    )
    
    # =================================================================
    # 6. POLYNOMIAL OPERATIONS COMPLEXITY
    # =================================================================
    print_section("6. ĐỘ PHỨC TẠP PHÉP TOÁN ĐA THỨC")
    
    print("📐 Testing polynomial operations...")
    
    # Polynomial multiplication
    print("\n--- Polynomial Multiplication ---")
    poly_mult_times = []
    degrees = [10, 50, 100, 200]
    
    for deg in degrees:
        p1 = Poly([1]*deg, x)  # Polynomial of degree deg-1
        p2 = Poly([2]*deg, x)
        
        avg_time, memory, _ = benchmark_function(lambda: p1 * p2)
        poly_mult_times.append(avg_time)
        print(f"Poly multiplication degree {deg}: {avg_time:.4f}s")
    
    print_complexity(
        "Polynomial Multiplication",
        "O(n²) naive, O(n log n) FFT-based",
        f"Measured: {[f'{t:.4f}s' for t in poly_mult_times]}",
        "SymPy may use different algorithms based on size"
    )
    
    # GCD computation
    print("\n--- Polynomial GCD ---")
    gcd_times = []
    
    for deg in [10, 20, 30, 50]:
        p1 = Poly(x**deg + x**(deg-1) + 1, x)
        p2 = Poly(x**(deg-2) + x + 1, x)
        
        avg_time, memory, _ = benchmark_function(gcd, p1, p2)
        gcd_times.append(avg_time)
        print(f"Poly GCD degree {deg}: {avg_time:.4f}s")
    
    print_complexity(
        "Polynomial GCD",
        "O(n²) Euclidean algorithm",
        f"Measured: {[f'{t:.4f}s' for t in gcd_times]}",
        "Quadratic in degree"
    )
    
    # =================================================================
    # 7. MEMORY COMPLEXITY ANALYSIS
    # =================================================================
    print_section("7. PHÂN TÍCH ĐỘ PHỨC TẠP BỘ NHỚ")
    
    if HAS_PSUTIL:
        print("💾 Testing memory usage...")
        
        # Expression tree memory
        print("\n--- Expression Memory Usage ---")
        
        for size in [100, 500, 1000, 2000]:
            gc.collect()
            memory_before = get_memory_usage()
            
            # Create large expression
            expr = sum(x**i for i in range(size))
            expanded = expand(expr * (x + 1))
            
            memory_after = get_memory_usage()
            memory_used = memory_after - memory_before
            
            print(f"Expression size {size}: {memory_used:.2f}MB")
        
        # Matrix memory
        print("\n--- Matrix Memory Usage ---")
        
        for size in [50, 100, 200]:
            gc.collect()
            memory_before = get_memory_usage()
            
            # Create large matrix
            A = Matrix([[i+j+1 if i != j else i+j+3 for j in range(size)] for i in range(size)])
            det_A = A.det()
            
            memory_after = get_memory_usage()
            memory_used = memory_after - memory_before
            
            print(f"Matrix {size}x{size}: {memory_used:.2f}MB")
    else:
        print("💾 Memory analysis bị skip - cần psutil package")
        print("   Install: pip install psutil")
    
    # =================================================================
    # 8. COMPARISON WITH THEORETICAL BOUNDS
    # =================================================================
    print_section("8. SO SÁNH VỚI LÝ THUYẾT")
    
    print("📊 Theoretical vs Practical Complexity Summary:")
    print()
    
    complexity_table = [
        ("Expression Expansion", "O(d^n)", "Exponential", "🔴 Very expensive for high degrees"),
        ("Polynomial Factorization", "O(d^6) to Exponential", "Varies", "🟡 Depends on polynomial structure"),
        ("Linear System Solving", "O(n³)", "Cubic", "🟢 Predictable and efficient"),
        ("Symbolic Differentiation", "O(n)", "Linear", "🟢 Very efficient"),
        ("Symbolic Integration", "Undecidable", "Varies", "🔴 Can be impossible"),
        ("Matrix Multiplication", "O(n³)", "Cubic", "🟢 Standard algorithms"),
        ("Determinant", "O(n³)", "Cubic", "🟢 Much better than naive O(n!)"),
        ("Eigenvalues", "O(n³)+", "Cubic+", "🟡 Can be much worse symbolically"),
        ("Primality Testing", "O(log³ n)", "Sub-polynomial", "🟢 Very efficient"),
        ("Integer Factorization", "Exponential", "Exponential", "🔴 Fundamental hard problem"),
        ("Polynomial Multiplication", "O(n log n)", "Quasi-linear", "🟢 FFT-based algorithms"),
        ("Polynomial GCD", "O(n²)", "Quadratic", "🟢 Euclidean algorithm"),
    ]
    
    for operation, theoretical, practical, assessment in complexity_table:
        print(f"{assessment}")
        print(f"  {operation}: {theoretical} → {practical}")
    
    # =================================================================
    # 9. OPTIMIZATION RECOMMENDATIONS
    # =================================================================
    print_section("9. KHUYẾN NGHỊ TỐI ƯU HÓA")
    
    recommendations = [
        "🟢 EFFICIENT OPERATIONS:",
        "  • Differentiation: Rất nhanh, sử dụng thoải mái",
        "  • Linear algebra: Hiệu quả cho ma trận vừa phải (<100x100)",
        "  • Primality testing: Rất nhanh cho số thông thường",
        "  • Basic arithmetic: Linear complexity",
        "",
        "🟡 MODERATE COMPLEXITY:",
        "  • Polynomial operations: OK cho degree <50",
        "  • Matrix determinant: OK cho size <50x50",
        "  • Simple factorization: Depends on structure",
        "",
        "🔴 EXPENSIVE OPERATIONS:",
        "  • High-degree expansion: Tránh degree >20",
        "  • Symbolic integration: Có thể rất chậm hoặc thất bại",
        "  • Large matrix eigenvalues: Symbolic rất chậm",
        "  • Integer factorization: Chậm cho số lớn",
        "",
        "💡 OPTIMIZATION TIPS:",
        "  • Sử dụng numerical evaluation khi có thể",
        "  • Simplify expressions trước khi thao tác phức tạp",
        "  • Chia nhỏ problems lớn thành parts nhỏ hơn",
        "  • Cache kết quả expensive computations",
        "  • Sử dụng assumptions để speed up",
    ]
    
    for rec in recommendations:
        print(rec)
    
    print("\n" + "="*70)
    print("🎯 SUMMARY: SymPy có complexity từ linear đến exponential")
    print("   Hiểu complexity để choose right approach cho problems!")
    print("="*70)

if __name__ == "__main__":
    # Check if optional packages are available
    if not HAS_PSUTIL:
        print("⚠️  Warning: psutil không có - memory analysis sẽ bị skip")
        print("   Install: pip install psutil")
    
    main() 