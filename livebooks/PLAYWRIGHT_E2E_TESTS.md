# Playwright E2E Tests for `bsts_nx_guide.livemd`

## Prerequisites

```bash
# 1. Start Livebook with a caller-supplied random password for Playwright access
export LIVEBOOK_PASSWORD="$(openssl rand -base64 24)"
livebook server --ip 127.0.0.1 --port 8080

# 2. Install Playwright
npm install playwright
npx playwright install chromium
```

**Note:** Livebook Desktop uses internal token auth that external browsers can't access.
For E2E testing, run a standalone Livebook server with `LIVEBOOK_PASSWORD` set.

## Code Validation Status

All 23 sections validated via ExUnit: **23/23 pass** (see `test/livebook_guide_validation_test.exs`).

| Section | Test | Time |
|---------|------|------|
| 1. Random walk | `section 1 - random walk generation` | 1ms |
| 3. Synthetic data | `section 3 - synthetic data generation` | <1ms |
| 4. Adstock + saturation | `section 4 - adstock and hill saturation` | <1ms |
| 5. Kalman filter | `section 5 - kalman filter` | 11ms |
| 6. RTS smoother | `section 6 - RTS smoother` | 4ms |
| 7.1 Scalar Gibbs | `section 7.1 - scalar gibbs sampler` | 646ms |
| 7.2 Multi-chain diagnostics | `section 7.2 - multiple chains and diagnostics` | 397ms |
| 7.3 Structured model | `section 7.3 - structured model composition` | 20s |
| 8.1 Scalar causal impact | `section 8.1 - scalar causal impact` | 277ms |
| 8.2 Structured causal impact | `section 8.2 - structured causal impact` | 5s |
| 8.4 Filter-based impact | `section 8.4 - filter-based causal impact` | 11ms |
| 9. InterventionAnalysis | `section 9 - intervention analysis one-call API` | 22s |
| 10. Shapley attribution | `section 10 - shapley attribution` | 5s |
| 11. Full pipeline | `section 11 - full pipeline` | 5s |
| 12. Marketing lift | `section 12 - marketing lift case study` | 548ms |
| 13. Demand forecasting | `section 13 - demand forecasting case study` | 2s |
| 14. Anomaly detection | `section 14 - anomaly detection case study` | 7ms |
| 15. Policy evaluation | `section 15 - policy evaluation case study` | 2s |
| 16. TV attribution | `section 16 - TV attribution wrapper` | 67s |
| 17. Covariate selection | `section 17 - covariate selection` | <1ms |
| 18. Validation | `section 18 - validation and calibration` | 67s |
| 19. Forecasting | `section 19 - forecasting` | 92s |
| 21. API reference | `section 21 - all modules export functions` | 3ms |

---

## Playwright Test Steps

### Test 1: Authentication

```
Step: Navigate to Livebook URL
Action: page.goto('http://localhost:8080')
Expect: Page loads without error

Step: Enter password
Action: page.locator('input[name="password"]').fill(LIVEBOOK_PASSWORD)
Action: page.locator('button:has-text("Authenticate")').click()
Expect: Redirected to home page (URL does not contain '/authenticate')
Screenshot: 01_authenticated.png
```

### Test 2: Open Notebook

```
Step: Open notebook file
Action: page.goto('http://localhost:8080/open/file?path=/Users/nicholas/develop/bsts_nx/livebooks/bsts_nx_guide.livemd')
Expect: page.waitForURL('**/sessions/**', { timeout: 10000 })
Expect: Page title contains 'From Noisy Data to Causal Claims'
Screenshot: 02_notebook_opened.png
```

### Test 3: Setup Cell Execution

```
Step: Locate the setup cell
Selector: page.locator('[data-el-session]')
Expect: Setup section visible

Step: Click "Setup" button (or "Reconnect and setup" if already connected)
Action: page.locator('button:has-text("Setup"), button:has-text("Reconnect and setup")').first().click()
Expect: Setup cell starts evaluating

Step: Wait for setup to complete
Action: Wait for Mix.install to finish — this can take 60-120s on first run
Selector: page.locator('[data-el-cell-status="evaluated"]').first()
Timeout: 180000 (3 minutes)
Expect: Setup cell shows evaluated status (green indicator)
Screenshot: 03_setup_complete.png
```

### Test 4: Evaluate All Cells

```
Step: Trigger "Evaluate all" from the notebook menu
Action: page.locator('[data-el-notebook-headline]').hover()
Action: page.keyboard.press('Meta+Shift+Enter')
  -- OR --
Action: page.locator('button[aria-label*="evaluate"]').click()
Expect: All cells begin evaluating (spinners appear)

Step: Wait for all evaluations to complete
Poll Interval: 15 seconds
Max Wait: 600 seconds (10 minutes)
Poll Check: count of evaluating cells
  Selector: page.locator('[data-el-cell][data-js-evaluating]').count()
  Exit When: count === 0
Screenshot: 04_all_evaluated.png (every 60s during wait)
```

### Test 5: Verify No Errors

```
Step: Count error outputs across all cells
Selector: page.locator('[data-el-outputs] .error, .cell-status-error').count()
Expect: count === 0
If Fail: Screenshot failing cells, extract error text:
  Action: page.locator('.error').allTextContents()
Screenshot: 05_no_errors.png
```

### Test 6: Verify Section Headings Present

```
Step: Check all major headings exist in the rendered notebook

Headings to verify:
  - "From Noisy Data to Causal Claims"
  - "Why raw data lies to you"
  - "The core idea: decompose, project, compare"
  - "A world where we know the truth"
  - "How marketing effects propagate"
  - "The Kalman filter: tracking a signal through noise"
  - "The smoother: the power of hindsight"
  - "The Gibbs sampler: honest about what we don't know"
  - "The counterfactual: what would have happened?"
  - "The one-call API"
  - "Attribution: splitting credit fairly"
  - "The full pipeline"
  - "Case Study: Did our email campaign drive conversions?"
  - "Case Study: How much ice cream should we stock?"
  - "Case Study: Is this server spike an anomaly?"
  - "Case Study: Did the new speed limit reduce accidents?"
  - "Case Study: Which TV spots earned their airtime?"
  - "Covariate selection"
  - "Breaking your own results"
  - "Bayesian forecasting with credible intervals"
  - "When to use what"
  - "API reference"
  - "Exercises for the curious"
  - "What you've learned"

Action per heading:
  page.locator('h1, h2').filter({ hasText: heading })
  Expect: count >= 1

Screenshot: 06_headings.png
```

### Test 7: Verify VegaLite Charts Rendered

```
Step: Count rendered VegaLite chart containers
Selector: page.locator('.vega-embed, canvas').count()
Expect: count >= 10 (notebook has ~15 charts)

Step: Verify specific charts by section
  Section 1 (random walk):
    Scroll to section, check canvas exists
  Section 3 (synthetic decomposition):
    Scroll to section, check chart with intervention line
  Section 4 (adstock):
    Scroll to section, check 3-series line chart
  Section 5 (Kalman filter):
    Scroll to section, check filter vs prediction chart
  Section 8.1 (causal impact band):
    Scroll to section, check band plot with actual overlay
  Section 8.2 (spaghetti plot):
    Scroll to section, check spaghetti plot renders
  Section 10 (attribution bar chart):
    Scroll to section, check bar chart
  Section 10 (gantt chart):
    Scroll to section, check spot window visualization
  Section 14 (anomaly z-scores):
    Scroll to section, check z-score threshold chart

Screenshot: 07_charts_verified.png
```

### Test 8: Verify Kino Outputs

```
Step: Check Kino.Markdown outputs render correctly
Selector: page.locator('[data-el-outputs] table').count()
Expect: count >= 5 (multiple markdown tables in the notebook)

Step: Check Kino.DataTable outputs render
Selector: page.locator('.kino-data-table, [data-el-outputs] .table-container').count()
Expect: count >= 2 (attribution data table, API reference table)

Step: Verify specific data tables
  Section 10 (attributions):
    Expect: DataTable with columns: spot_id, lift, ci_lower, ci_upper
  Section 17 (covariate selection):
    Expect: DataTable with columns: candidate, correlation, selected?
  Section 21 (API reference):
    Expect: DataTable with columns: module, function

Screenshot: 08_kino_outputs.png
```

### Test 9: Verify Intervention Report

```
Step: Find the InterventionAnalysis report output (Section 9)
Scroll: Navigate to "The one-call API" section
Selector: page.locator('[data-el-outputs]').filter({ hasText: 'Significant' })
Expect: Output contains "Significant?" with "Yes" or "No"
Expect: Output contains "Cumulative effect"
Expect: Output contains "Relative effect"

Step: Verify the full text report renders
Selector: Look for the InterventionAnalysis.report() output
Expect: Contains multi-line intervention analysis text

Screenshot: 09_intervention_report.png
```

### Test 10: Verify Mathematical Content

```
Step: Check LaTeX/KaTeX renders in markdown cells
Selector: page.locator('.katex, .math, mjx-container').count()
Expect: count >= 3 (state-space equations, Shapley formula, adstock formula)

Step: Verify specific equations
  Section 4 (adstock): Check adstock(t) equation renders
  Section 5 (state-space): Check state evolution equations render
  Section 10 (Shapley): Check Shapley value formula renders

Screenshot: 10_math_content.png
```

### Test 11: Full Scroll-Through Visual Check

```
Step: Scroll from top to bottom, capturing screenshots every section

Action: Iterate through all h2 elements
  For each heading:
    Action: heading.scrollIntoViewIfNeeded()
    Action: page.waitForTimeout(1000)
    Screenshot: section_{index}_{heading_slug}.png

Expect: No blank/broken output sections
Expect: No JavaScript console errors
  Check: page.on('console', msg => { if (msg.type() === 'error') fail() })

Screenshot: 11_scroll_complete.png
```

### Test 12: Notebook Re-evaluation (Idempotency)

```
Step: After full evaluation, re-run all cells
Action: page.keyboard.press('Meta+Shift+Enter')

Step: Wait for re-evaluation to complete
Action: Same polling as Test 4

Step: Verify no new errors
Selector: Same as Test 5
Expect: count === 0 (notebook is idempotent)

Screenshot: 12_re_evaluation.png
```

---

## Test Runner Script Skeleton

```javascript
// test_bsts_guide.spec.js
const { test, expect } = require('@playwright/test');

const LIVEBOOK_URL = process.env.LIVEBOOK_URL || 'http://localhost:8080';
const LIVEBOOK_PASSWORD = process.env.LIVEBOOK_PASSWORD;
if (!LIVEBOOK_PASSWORD) {
  throw new Error('LIVEBOOK_PASSWORD must be set for Playwright Livebook auth');
}
const NOTEBOOK_PATH = '/Users/nicholas/develop/bsts_nx/livebooks/bsts_nx_guide.livemd';

test.describe('BSTS Guide Livebook', () => {
  test.setTimeout(900_000); // 15 min total

  test('authenticates and opens notebook', async ({ page }) => {
    await page.goto(LIVEBOOK_URL);
    if (page.url().includes('/authenticate')) {
      await page.locator('input[name="password"]').fill(LIVEBOOK_PASSWORD);
      await page.locator('button:has-text("Authenticate")').click();
      await page.waitForURL('**/', { timeout: 10_000 });
    }
    await page.goto(`${LIVEBOOK_URL}/open/file?path=${encodeURIComponent(NOTEBOOK_PATH)}`);
    await page.waitForURL('**/sessions/**', { timeout: 15_000 });
    await expect(page).toHaveTitle(/From Noisy Data/);
  });

  test('setup cell completes without error', async ({ page }) => {
    // ... authenticate and open first ...
    const setupBtn = page.locator('button:has-text("Setup")').first();
    if (await setupBtn.isVisible()) await setupBtn.click();
    // Wait for setup to complete (Mix.install)
    await page.waitForSelector('[data-el-cell-status="evaluated"]', { timeout: 180_000 });
    const errors = await page.locator('.error').count();
    expect(errors).toBe(0);
  });

  test('all cells evaluate without errors', async ({ page }) => {
    // ... authenticate, open, setup first ...
    await page.keyboard.press('Meta+Shift+Enter');
    // Poll until no cells are evaluating
    await expect(async () => {
      const evaluating = await page.locator('[data-el-cell][data-js-evaluating]').count();
      expect(evaluating).toBe(0);
    }).toPass({ timeout: 600_000, intervals: [15_000] });
    const errors = await page.locator('.error').count();
    expect(errors).toBe(0);
  });

  test('charts are rendered', async ({ page }) => {
    const charts = await page.locator('.vega-embed, canvas').count();
    expect(charts).toBeGreaterThanOrEqual(10);
  });

  test('all sections present', async ({ page }) => {
    const sections = [
      'Why raw data lies', 'The Kalman filter', 'The Gibbs sampler',
      'The counterfactual', 'Attribution', 'Did our email campaign',
      'How much ice cream', 'Is this server spike', 'Did the new speed limit',
      'Which TV spots', 'Bayesian forecasting', 'API reference'
    ];
    for (const s of sections) {
      await expect(page.locator('h1, h2').filter({ hasText: s })).toHaveCount(1);
    }
  });
});
```

## Running the Tests

```bash
# Start Livebook
export LIVEBOOK_PASSWORD="$(openssl rand -base64 24)"
livebook server --ip 127.0.0.1 --port 8080 &

# Run Playwright tests
npx playwright test test_bsts_guide.spec.js --headed

# Or headless with screenshots
npx playwright test test_bsts_guide.spec.js --reporter=html
```
