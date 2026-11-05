# Adaptive Change Point Detection in the S and P 500

Bachelor thesis project, IIT Kharagpur. Based on the paper by Musaev, Grigoriev and Kolosov, "Adaptive algorithms for change point detection in financial time series", AIMS Mathematics, 2024.

## The problem in one line

Prices move all the time. Can we tell, soon after it happens and using only the past, that the market has really changed direction, and not just wobbled?

## Why this is hard

- If we react quickly, normal wobbles cause false alarms.
- If we wait to be sure, we are late.
- Market behaviour keeps changing, so one fixed setting may not suit every period.

That last point is the idea behind "adaptive". An adaptive method changes its own settings as new data arrives. **Our main question is: does adapting actually do better than one well chosen fixed setting?**

## Where this is useful in real life

- **Investing and risk control:** reduce exposure sooner after a market turns down.
- **Banks and regulators:** notice when market stress begins.
- **Other fields, for future work:** machine health, patient monitoring, climate and earthquake records. The same maths applies to any data recorded over time.

This project does not give trading advice, and it does not test trading profit.

## Why this belongs in mathematics

| Maths topic | Where it appears |
|---|---|
| Time series | Prices are modelled as trend plus noise |
| Recursive estimation | Smoothing is a weighted average with weights that shrink geometrically |
| Least squares | Fitting straight lines and regression |
| Hypothesis testing | Is a change real, or just noise? |
| Sequential analysis | CUSUM comes from a likelihood ratio test |
| Non parametric statistics | Wilcoxon test, bootstrap intervals, random alarm test |

We can also prove the central trade off. With smoothing factor alpha, a steady trend with slope b leaves the line behind by b(1 - alpha)/alpha, while noise variance shrinks by alpha/(2 - alpha). Lower alpha means less noise and more delay. No choice of alpha removes both. The full working is in the report.

## The data

- **Main series:** S and P 500 index, daily closing price, using the log of the price, so a 10 percent move looks the same in any year.
- **Related series:** 20 large US companies, needed only for Method 4.
- **Dates:** 2005 to 2022. Settings are chosen on 2005 to 2014 and tested once on 2015 to 2022.
- **Where it comes from:** the data files of the open source skfolio package, saved in `data/sp500_bundled.csv`, so the project runs without internet.
- **Fresh data to 2025:** an option downloads the S and P 500, Nasdaq, Dow Jones and Russell 2000 from Yahoo Finance and tests on 2016 to 2025. The results below will change if you use it, and the wording of the report would then need updating to match.
- **Later:** to add another market, put a new CSV in the `data` folder with a price column for the target and one column for each related series.

## The five methods, plus one control

| # | Method | Type | In plain words | File |
|---|---|---|---|---|
| 1 | Fixed smoothing | Fixed | Smooth the price with one constant alpha. Alarm when the line reverses by a set amount. | `src/methods/method_1_fixed_smoothing.py` |
| 2 | Adaptive smoothing | Adaptive | Same, but alpha rises when surprises keep pointing the same way. | `src/methods/method_2_adaptive_smoothing.py` |
| 3 | Adaptive window | Adaptive | Fit a straight line to recent days. Shrink the window after a big surprise, and grow it when calm. | `src/methods/method_3_adaptive_window.py` |
| 4 | Regression | Adaptive | Predict the target from its most related series. Alarm when the gap gets too big. | `src/methods/method_4_regression.py` |
| 5 | CUSUM | Benchmark | Add up unusual daily returns. Alarm when the total crosses a limit. | `src/methods/method_5_cusum.py` |
| Control | Fixed window | Fixed | Method 3 with a window that never changes, so Method 3 has a fair comparison. | same file as Method 3 |

Method 4 looks for a broken relationship between series, not a change in market direction. It is therefore not asking quite the same question as the others.

A small worked example of Method 1 with alpha 0.1. Start at 10. A new value of 20 gives 10 + 0.1 x (20 - 10) = 11. Another 20 gives 11 + 0.1 x (20 - 11) = 11.9. The line moves slowly toward the new level, which removes noise but adds delay.

## Workflow

1. **Prepare data.** Load prices and take logs.
2. **Test on fake markets.** Here we planted the changes, so we know the right answers.
   - Case A: up and down trends with calm noise.
   - Case B: the same trends with wild noise.
   - Case C: the target leaves and rejoins its related series.
3. **Test on the S and P 500.** Real markets have no official answer sheet, so we build one with hindsight. A high or low counts as a turning point if the price later reverses by at least 10 percent. This answer key is used only for scoring.
4. **Score with one set of rules.** An alarm is correct if it comes within 40 trading days after a true change. We report F1, a single score from 0 to 1 that rewards catching changes and punishes false alarms.
5. **Train a machine learning model.** The Colab notebook combines all the signals, to see if a model beats the simple rules.

## What we found

**Fake markets, average F1 on 200 fresh runs**

| Method | Case A | Case B | Case C |
|---|---:|---:|---:|
| Fixed smoothing | 0.788 | 0.620 | 0.310 |
| Adaptive smoothing | 0.789 | 0.644 | 0.309 |
| Fixed window control | 0.780 | 0.631 | 0.318 |
| Adaptive window | 0.776 | 0.601 | 0.314 |
| Regression | 0.238 | 0.245 | 0.338 |
| CUSUM | 0.327 | 0.258 | 0.286 |

**S and P 500, test years 2015 to 2022, 18 turning points**

| Method | Alarms | Correct | F1 | Average delay, days | Better than random alarms? |
|---|---:|---:|---:|---:|---|
| Fixed smoothing | 16 | 13 | 0.765 | 17.5 | Yes |
| Adaptive smoothing | 20 | 14 | 0.737 | 15.2 | Yes |
| Fixed window control | 14 | 11 | 0.688 | 17.5 | Yes |
| Adaptive window | 14 | 11 | 0.688 | 19.9 | Yes |
| Regression | 34 | 11 | 0.423 | 13.1 | No clear evidence |
| CUSUM | 40 | 12 | 0.414 | 8.4 | No clear evidence |

**Machine learning on the same test years:** gradient boosting scored 0.500 and logistic regression 0.444, both below the fixed smoothing rule.

**What this means**

1. **Adapting was not automatically better.** With calm noise it made no real difference. With wild noise, adaptive smoothing gained a little (0.644 against 0.620) while the adaptive window lost a little (0.601 against 0.631).
2. **On the real test years, fixed smoothing scored highest.** Adaptive smoothing caught one more turning point and reacted about 2 days sooner, but raised more false alarms. With only 18 turning points, this gap is too small to name a true winner, and the order changes under other scoring rules.
3. **Each method suits its own kind of change.** Trend methods did well on trend reversals. Regression came first on relationship breaks, though only by a small margin.
4. **More complexity did not help.** The machine learning model did not beat a simple rule. It did carry real information, as it did better than guessing.

**Limits to be open about**

- There is only one real market history, with 36 turning points in total.
- The answer key has a similar up and down logic to the trend methods, which favours them over CUSUM and regression.
- The 20 companies are part of the index and were picked from a recent list, so Method 4 has a built in link and some survivorship bias.
- This project uses some ideas from the paper and not all of them. The two way filter and the historical pattern matching method are left for future work.

## How to run it

**You need:** Python 3.10 or newer. Check by typing `python --version` in a terminal.

**Step 1.** Download this project from GitHub using Code, then Download ZIP. Unzip it.

**Step 2.** Open a terminal inside the project folder.

**Step 3.** Install the tools the project needs. You only do this once.

```
pip install -r requirements.txt
```

**Step 4.** Run everything. It takes about 2 to 3 minutes.

```
python run_all.py
```

**Step 5.** Look at the results. Tables are in the `results` folder, and charts are in `results/figures`.

**Optional: run one part at a time**

```
python tests/test_methods.py
python step1_prepare_data.py
python step2_run_simulation.py
python step3_run_sp500.py
python step4_make_figures.py
python evaluate.py
```

**Optional: use fresh data to 2025**

```
python run_all.py --source yahoo
```

**Machine learning model:** open `notebooks/Adaptive_change_point.ipynb` in Google Colab, then choose Runtime, then Run all.

If `python` is not found, try `python3`. On some computers, `pip` must also be written as `pip3`.

## Project map

```
config.py                 all settings in one place
step1_prepare_data.py     load the data
step2_run_simulation.py   test on fake markets
step3_run_sp500.py        test on the S and P 500
step4_make_figures.py     charts and report tables
evaluate.py               scoring rules, kept separate from the methods
run_all.py                runs every step
src/methods/              one file for each method
src/detector.py           the direction flip alarm rule
src/simulate.py           the fake markets
tests/test_methods.py     checks the maths and that no method sees the future
notebooks/                machine learning notebook for Colab
report/                   thesis report in LaTeX
```

## How to explain it to the panel

"Prices are trend plus noise. Separating them forces a trade off between speed and false alarms, and I can show that with a formula. Adaptive methods try to escape this by changing their settings as data arrives. I tested this fairly, first where the truth is known and then on the S and P 500, using settings chosen only from earlier years. Adapting did not reliably beat one good fixed setting. So the useful result is knowing when adapting helps, and how little it helped here, not simply that a complex method wins."

## Main references

- Musaev, Grigoriev, Kolosov. Adaptive algorithms for change point detection in financial time series. AIMS Mathematics 9(12), 2024.
- Page. Continuous inspection schemes. Biometrika 41, 1954.
- Trigg, Leach. Exponential smoothing with an adaptive response rate. Operational Research Quarterly 18, 1967.
