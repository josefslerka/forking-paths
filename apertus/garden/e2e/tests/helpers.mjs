import {expect, test as base} from "@playwright/test";

export const test = base.extend({
  appPage: async ({page}, use) => {
    const clientErrors = [];

    page.on("pageerror", (error) => {
      clientErrors.push(`pageerror: ${error.message}`);
    });
    page.on("console", (message) => {
      if (message.type() === "error") {
        clientErrors.push(`console.error: ${message.text()}`);
      }
    });

    await use(page);

    expect.soft(
      clientErrors,
      "Stránka během uživatelské cesty nesmí vyhazovat klientské chyby."
    ).toEqual([]);
  }
});

export {expect};

export async function expectWorkbenchReady(page, {mock, timeout = 30_000}) {
  await expect.poll(
    async () => {
      const response = await page.request.get("/api/health");
      if (!response.ok()) return `http:${response.status()}`;
      const health = await response.json();
      return `${health.status}:${String(health.mock)}`;
    },
    {
      message: mock
        ? "Deterministický stub se nepřepnul do stavu ready."
        : "Živý Apertus se nepřepnul do stavu ready.",
      timeout
    }
  ).toBe(`ready:${String(mock)}`);
}

export async function openWorkbench(page, path, options = {}) {
  const response = await page.goto(path, {waitUntil: "domcontentloaded"});
  expect(response, `Navigace na ${path} musí vrátit HTTP odpověď.`).not.toBeNull();
  expect(response.ok(), `Navigace na ${path} selhala HTTP ${response.status()}.`).toBeTruthy();
  await expect(page.locator("main")).toBeVisible();
  if (options.mock !== undefined) {
    await expectWorkbenchReady(page, options);
  }
}

export async function expectFiniteText(locator) {
  await expect(locator).toBeVisible();
  await expect(locator).not.toHaveText(/^(?:—|\s*)$/);
  const value = (await locator.textContent())?.trim() || "";
  expect(value).not.toMatch(/\b(?:NaN|Infinity|undefined|null)\b/);
}

export async function expectNoBusyOverlay(page) {
  const overlay = page.locator("#busy-overlay");
  if (await overlay.count()) {
    await expect(overlay).toBeHidden();
  }
}

export async function expectReadableText(locator, minimum = 4.5) {
  const failures = await locator.evaluateAll((elements, threshold) => {
    const rgba = (value) => {
      const parts = value.match(/[\d.]+/g).map(Number);
      return [parts[0], parts[1], parts[2], parts[3] ?? 1];
    };
    const over = (foreground, background) => foreground.slice(0, 3).map(
      (channel, i) => channel * foreground[3] + background[i] * (1 - foreground[3])
    ).concat(1);
    const backgrounds = (element) => {
      if (!element) return [[255, 255, 255, 1]];
      const style = getComputedStyle(element);
      const base = backgrounds(element.parentElement).map(
        (background) => over(rgba(style.backgroundColor), background)
      );
      const stops = style.backgroundImage.match(/rgba?\([^)]*\)/g) ?? [];
      return stops.length
        ? stops.flatMap((stop) => base.map((background) => over(rgba(stop), background)))
        : base;
    };
    const luminance = (color) => {
      const linear = color.slice(0, 3).map((channel) => {
        const value = channel / 255;
        return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
      });
      return linear[0] * 0.2126 + linear[1] * 0.7152 + linear[2] * 0.0722;
    };
    return elements.flatMap((element) => {
      const foreground = rgba(getComputedStyle(element).color);
      const contrast = Math.min(...backgrounds(element).map((background) => {
        const a = luminance(over(foreground, background));
        const b = luminance(background);
        return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
      }));
      return contrast >= threshold ? [] : [{
        text: element.textContent.trim(),
        element: element.className,
        contrast: Number(contrast.toFixed(2))
      }];
    });
  }, minimum);
  expect(failures, `Text musí mít kontrast alespoň ${minimum}:1 vůči své ploše.`).toEqual([]);
}
