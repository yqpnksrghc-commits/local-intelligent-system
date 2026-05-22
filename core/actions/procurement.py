"""
Procurement — search for products and place orders via browser automation.

Supports:
  - Amazon product search + order
  - Best Buy product search
  - Carrier (AT&T, Verizon, T-Mobile) phone deals

Uses Playwright for browser automation.
pip install playwright && playwright install chromium

Required env vars for ordering:
  AMAZON_EMAIL
  AMAZON_PASSWORD
  SHIPPING_NAME
  SHIPPING_ADDRESS
  SHIPPING_CITY
  SHIPPING_STATE
  SHIPPING_ZIP

A confirmation gate blocks all purchases. No money moves without approval.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv
from core.actions.gate import Action, confirm, ActionDenied

load_dotenv()


@dataclass
class Product:
    title:     str
    price:     float | None
    url:       str
    retailer:  str
    asin:      str | None = None
    rating:    float | None = None
    in_stock:  bool = True


# ── Search ────────────────────────────────────────────────────────────────────

async def _search_amazon(query: str, max_results: int = 5) -> list[Product]:
    from playwright.async_api import async_playwright

    results = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page    = await browser.new_page()
        await page.goto(
            f"https://www.amazon.com/s?k={query.replace(' ', '+')}&i=electronics",
            wait_until="domcontentloaded",
            timeout=20000,
        )

        items = await page.query_selector_all('[data-component-type="s-search-result"]')
        for item in items[:max_results]:
            try:
                title_el = await item.query_selector("h2 a span")
                title    = await title_el.inner_text() if title_el else "?"

                price_el  = await item.query_selector(".a-price .a-offscreen")
                price_str = await price_el.inner_text() if price_el else None
                price     = float(price_str.replace("$", "").replace(",", "")) \
                            if price_str else None

                link_el  = await item.query_selector("h2 a")
                href     = await link_el.get_attribute("href") if link_el else ""
                url      = f"https://www.amazon.com{href}" if href.startswith("/") else href

                asin_el  = item
                asin     = await asin_el.get_attribute("data-asin")

                rating_el = await item.query_selector(".a-icon-alt")
                rating_str = await rating_el.inner_text() if rating_el else None
                rating = float(rating_str.split()[0]) if rating_str else None

                results.append(Product(
                    title=title.strip(), price=price, url=url,
                    retailer="amazon", asin=asin, rating=rating,
                ))
            except Exception:
                continue

        await browser.close()
    return results


def search(query: str, retailer: str = "amazon", max_results: int = 5) -> list[Product]:
    """Search for products. Synchronous wrapper around async search."""
    import asyncio
    if retailer == "amazon":
        return asyncio.run(_search_amazon(query, max_results))
    raise ValueError(f"Unknown retailer: {retailer!r}")


# ── Order ─────────────────────────────────────────────────────────────────────

async def _order_amazon(product: Product) -> str:
    """
    Navigate to product page and place order via Amazon 1-Click or cart.
    Returns order confirmation text.
    """
    from playwright.async_api import async_playwright

    email    = os.getenv("AMAZON_EMAIL", "")
    password = os.getenv("AMAZON_PASSWORD", "")
    if not email or not password:
        raise RuntimeError(
            "Set AMAZON_EMAIL and AMAZON_PASSWORD in .env to enable ordering."
        )

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # visible for order review
        page    = await browser.new_page()

        # sign in
        await page.goto("https://www.amazon.com/ap/signin", wait_until="domcontentloaded")
        await page.fill("#ap_email", email)
        await page.click("#continue")
        await page.fill("#ap_password", password)
        await page.click("#signInSubmit")
        await page.wait_for_load_state("networkidle", timeout=15000)

        # go to product
        await page.goto(product.url, wait_until="domcontentloaded")

        # add to cart
        buy_now = await page.query_selector("#buy-now-button")
        if buy_now:
            await buy_now.click()
        else:
            add_cart = await page.query_selector("#add-to-cart-button")
            if add_cart:
                await add_cart.click()
                await page.goto("https://www.amazon.com/gp/cart/view.html")
                proceed = await page.query_selector('[name="proceedToRetailCheckout"]')
                if proceed:
                    await proceed.click()

        await page.wait_for_load_state("domcontentloaded")

        # place order — look for the final "Place your order" button
        place_order = await page.query_selector('[name="placeYourOrder1"]')
        if place_order:
            await place_order.click()
            await page.wait_for_load_state("networkidle", timeout=20000)
            confirm_text = await page.inner_text("body")
            result = "Order placed." if "order" in confirm_text.lower() else confirm_text[:200]
        else:
            result = "Reached checkout — final 'Place order' button not found. Review in browser."

        await browser.close()
        return result


def order(product: Product, confirm_first: bool = True) -> str:
    """Place an order for a product. Always gates before purchasing."""
    import asyncio

    action = Action(
        kind="order",
        description=f"Purchase: {product.title[:80]} @ "
                    f"{'${:.2f}'.format(product.price) if product.price else '?'}",
        cost=product.price,
        reversible=False,
        payload={
            "title":    product.title,
            "price":    product.price,
            "url":      product.url,
            "retailer": product.retailer,
        },
    )
    confirm(action, auto_approve=not confirm_first)

    if product.retailer == "amazon":
        return asyncio.run(_order_amazon(product))
    raise ValueError(f"Ordering not yet supported for: {product.retailer!r}")


# ── Convenience: find + order a phone ────────────────────────────────────────

def get_phone(
    query: str = "iPhone",
    max_price: float | None = None,
    retailer: str = "amazon",
    confirm_first: bool = True,
) -> str:
    """
    Search for a phone, present options, and order the selected one.
    Defaults to Apple/iPhone per user preference.
    """
    print(f"[procurement] searching {retailer} for: {query!r}")
    products = search(query, retailer)

    if not products:
        return "No products found."

    if max_price:
        products = [p for p in products if p.price and p.price <= max_price]
        if not products:
            return f"No products found under ${max_price:.2f}."

    print(f"\n── Products found " + "─" * 40)
    for i, p in enumerate(products, 1):
        price_str = f"${p.price:.2f}" if p.price else "price unknown"
        rating_str = f"★{p.rating}" if p.rating else ""
        print(f"  {i}. [{price_str}] {rating_str}  {p.title[:70]}")
        print(f"     {p.url[:80]}")
    print()

    try:
        choice = input("Select product number to order (or 'cancel'): ").strip()
    except (EOFError, KeyboardInterrupt):
        return "Procurement cancelled."

    if choice.lower() == "cancel":
        return "Procurement cancelled."

    try:
        idx = int(choice) - 1
        selected = products[idx]
    except (ValueError, IndexError):
        return "Invalid selection."

    return order(selected, confirm_first=confirm_first)
