import Stripe from "stripe";

export async function onRequestPost({ request, env }) {
  try {
    if (!env.STRIPE_SECRET_KEY) return new Response("Missing env.STRIPE_SECRET_KEY", { status: 500 });
    if (!env.STRIPE_WEBHOOK_SECRET) return new Response("Missing env.STRIPE_WEBHOOK_SECRET", { status: 500 });

    const stripe = new Stripe(env.STRIPE_SECRET_KEY);

    // 1) Read RAW body (required for signature verification)
    const body = await request.text();

    // 2) Grab signature header
    const sig = request.headers.get("stripe-signature");
    if (!sig) return new Response("Missing stripe-signature header", { status: 400 });

    // 3) Verify + construct event
    let event;
    try {
      event = stripe.webhooks.constructEvent(body, sig, env.STRIPE_WEBHOOK_SECRET);
    } catch (e) {
      return new Response(`Webhook signature verification failed: ${e.message}`, { status: 400 });
    }

    // 4) Only unlock on checkout completion
    if (event.type === "checkout.session.completed") {
      const session = event.data.object;

      const mapSessionId = session?.metadata?.map_session_id;
      if (!mapSessionId) {
        return new Response("Missing map_session_id in session metadata", { status: 400 });
      }

      // Save entitlement to D1
      // Requires D1 binding name "DB" (you have that)
      await env.DB.prepare(
        `INSERT INTO entitlements (map_session_id, unlocked, stripe_session_id, paid_at)
         VALUES (?, 1, ?, datetime('now'))
         ON CONFLICT(map_session_id) DO UPDATE SET
           unlocked=1,
           stripe_session_id=excluded.stripe_session_id,
           paid_at=excluded.paid_at`
      )
        .bind(mapSessionId, session.id)
        .run();
    }

    return new Response("ok", { status: 200 });
  } catch (err) {
    return new Response(`stripe-webhook error:\n${err?.message || String(err)}`, { status: 500 });
  }
}
