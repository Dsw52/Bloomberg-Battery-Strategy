# Appendix: AI Collaboration

This appendix documents how AI assistance was used during the assessment.

---

## 3.1 Prompts Used in Part One

### Prompt 1
> **User Prompt:** "Before applying any fix, distinguish between legitimate negative prices — which are a real structural oversupply signal and must be kept as-is — and telemetry placeholder errors like -9999 in rt_price, which should be treated as missing data. Count each category separately."

* **Why it worked:** Naming the two distinct failure modes explicitly, and forbidding the AI from collapsing them into a single "clean negative values" step, prevented a damaging cleaning mistake that may have occurred if the code silently deleted or averaged away real economic grid signals.

### Prompt 2
> **User Prompt:** "Don't use linear interpolation to fill missing hours or nulls — use a seasonal fill based on the same hour-of-day and same weekday in a prior period, since price and load are strongly diurnal and weekly, not smooth."

* **Why it worked:** Specifying the mathematical shape of the fallback logic, rather than just asking to "fill missing values," steered the AI away from an incorrect choice of using `.interpolate()`, which would have flattened sharp evening price spikes into a non-physical straight-line ramp.

### Prompt 3
> **User Prompt:** "Your proposed dispatch logic is flawed because it allows the battery to discharge power in the morning before it has physically charged overnight. Rewrite the ranking mask to ensure the asset maintains chronological causality and respects state-of-charge constraints across the daily horizon."

* **Why it worked:** This prompt represents a correction where I challenged the AI's math. The AI had treated the 24-hour day as an independent pool of hours rather than a continuous timeline, resulting in physically impossible operations. Forcing chronological constraints ensured the model conformed to physical asset realities.

---

## 3.2 A Technical Failure — Round‑Trip Efficiency Directional Bias

### What Happened
During the initial implementation of the battery’s 85% round‑trip efficiency (RTE), the AI attempted to split losses equally across charging and discharging using:

$$\eta_{\text{charge}} = \eta_{\text{discharge}} = \sqrt{0.85} \approx 0.9219$$

However, the AI applied this efficiency incorrectly. It multiplied the battery’s charging power by the efficiency factor:

>df["charging_energy_drawn"] = POWER_CAPACITY_MW * ETA_CHARGE

This implied the battery only needed $\sim92.2\text{ MW}$ of grid power to store $100\text{ MW}$ of power, which is impossible. The AI misunderstood the direction of efficiency losses. To store $100\text{ MW}$ of power, the battery must draw:$$\frac{100}{0.9219} \approx 108.5\text{ MW}$$By multiplying instead of dividing, the AI created a battery that produced more net energy than it consumed which would violate basic energy‑balance principles. Because the script executed flawlessly without throwing runtime errors, this structural flaw would have remained invisible without a deliberate audit. I caught the error by verifying that annual aggregate discharge energy did not exceed aggregate charge energy.

I corrected the charging and discharging logic to enforce proper physical boundaries:
>df["charging_cost"] = (POWER_CAPACITY_MW / ETA_CHARGE) * df["da_price"]
df["discharging_revenue"] = (POWER_CAPACITY_MW * ETA_DISCHARGE) * df["da_price"]

This correction restored a physically realistic energy profile and reduced the overstated profit to its value of $4.25 million.


## 3.3 AI Error Check & Code Audit

### Areas of Immediate Concern

The submitted script compounds multiple critical flaws into a final financial figure that is structurally incorrect. First, it loads a different dataset (`market_data_2024.csv`) that fails to match the assigned 2025 target year, meaning every downstream metric describes the wrong market universe. I would want to look into this and structure the code to use the correct data. Second, it executes a blind data cleanup using `df.dropna()`, which does not provide visibility into how many records were dropped or whether the missingness was random or systematic. I would be interested in understanding what the data would look like after this function is completed.

### Major Issues
The dispatch model violates basic market mechanisms and physics. Firstly, the DA-RT spread sign convention is arithmetically inverted ($RT - DA$ instead of $DA - RT$), completely reversing the market's premium signal. It also optimizes a day-ahead strategy using Real-Time prices (`rt_price`), which are highly volatile and unknown in advance. Because it picks the absolute cheapest and most expensive hours out of a full day ex-post, it simulates perfection rather than an executable strategy. Furthermore, the model assumes a physically impossible, lossless asset by completely omitting the specified 85% round-trip efficiency penalty. Finally, it commits a fundamental dimensional unit error by multiplying price directly by power capacity ($\text{MW}$) rather than energy capacity ($\text{MWh}$), hardcoding the asset's duration constraint into a pandas slice rather than linking dynamic structural parameters.

---

### Commercial & Modeling Consequences

These errors render the script's output inaccurate for business planning or client delivery. Utilizing the wrong dataset and hidden sample sizes creates immediate professional credibility and diligence risks. Because the algorithm ignores real-world efficiency losses and optimizes against historical real-time pricing with perfect foresight, the resulting profit figure is heavily overstated. This output should be withheld from client presentations and investment committees until the file source, data-loss tracking, market basis, efficiency factors, and energy dimensions are completely corrected and independently re-validated.