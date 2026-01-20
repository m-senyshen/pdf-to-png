export async function onRequestPost(context) {
  const { request, env } = context;

  const STRIPE_SECRET_KEY = env.STRIPE_SECRET_KEY;
  const PRICE_CENTS_CAD = env.PRICE_CENTS_CAD || "1200"; // $12.00 default

  if (!STRIPE_SECRET_KEY) {
    return new Response("Missing STRIPE_SECRET_KEY", { status: 500 });
  }

  let body;
  try {
    body = await request.json();
  } catch {
    body = {};
  }

  const workSessionId = body.workSessionId || "unknown";
  const pointCount = String(body.pointCount || "");

  // Use the current site origin so it works on both preview + production
  const url = new URL(request.url);
  const origin = `${url.protocol}//${url.host}`;

  // Stripe success/cancel URLs (success includes session_id placeholder)
  const successUrl = `${origin}/app.html?session_id={CHECKOUT_SESSION_ID}`;
  const cancelUrl = `${origin}/app.html`;

  // Create Checkout Session via Stripe REST API (form-encoded)
  const params = new URLSearchParams();
  params.set("mode", "payment");
  params.set("success_url", successUrl);
  params.set("cancel_url", cancelUrl);

  // line item: “Map Extract — Export Points”
  params.set("line_items[0][quantity]", "1");
  params.set("line_items[0][price_data][currency]", "cad");
  params.set("line_items[0][price_data][unit_amount]", PRICE_CENTS_CAD);
  params.set("line_items[0][price_data][product_data][name]", "Map Extract — Export Points (per session)");

  // useful metadata
  params.set("metadata[work_session_id]", workSessionId);
  if (pointCount) params.set("metadata[point_count]", pointCount);

  // Create session
  const resp = await fetch("https://api.stripe.com/v1/checkout/sessions", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${STRIPE_SECRET_KEY}`,
      "Content-Type": "application/x-www-form-urlencoded"
    },
    body: params.toString()
  });

  const data = await resp.json();

  if (!resp.ok) {
    return new Response(JSON.stringify(data), {
      status: resp.status,
      headers: { "Content-Type": "application/json" }
    });
  }

  // Return the hosted URL (and id if you want it)
  return new Response(JSON.stringify({ id: data.id, url: data.url }), {
    status: 200,
    headers: {
      "Content-Type": "application/json",
      "Access-Control-Allow-Origin": "*"
    }
  });
}
