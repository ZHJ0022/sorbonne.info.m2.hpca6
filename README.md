# sorbonne.info.m2.hpca6

Project 6: Hull-White Monte Carlo simulation.

Question 1

Estimate zero-coupon bond prices and forward rates on [0,10].

Requires `nvcc` with C++17 support, `make`, and an NVIDIA GPU.

Run from the project directory:
```bash
make
./question1
```

Output: `question1.csv`.
Columns `T,P,f`: maturity, zero coupon bond price, and forward rate.

Compare the CSV with the analytical model (requires Python 3 and Matplotlib):
```bash
python3 scripts/analyze_question1.py
```

Outputs: `results/question1.png` and `results/question1_summary.txt`.

Clean the executable with `make clean`.

