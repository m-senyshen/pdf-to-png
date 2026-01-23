import Stripe from "stripe";

export async function onRequestPost({ request, env }) {
  try {
    if (!env.STRIPE_SECRET_KEY) {
      return new Response("Missing env.STRIPE_SECRET_KEY", { status: 500 });
    }
    if (!env.STRIPE_WEBHOOK_SECRET) {
      return new Response("Missing env.STRIPE_WEBHOOK_SECRET", { status: 500 });
    }
    if (!env.DB) {
      return new Response("Missing D1 binding env.DB", { status: 500 });
    }

    const stripe = new Stripe(env.STRIPE_SECRET_KEY);

    // IMPORTANT: Use the raw body for signature verification
    const payload = await request.text();
    const sig = request.headers.get("stripe-signature");
    if (!sig) return new Response("Missing Stripe-Signature header", { status: 400 });

    let event;
    try {
      event = stripe.webhooks.constructEvent(payload, sig, env.STRIPE_WEBHOOK_SECRET);
    } catch (e) {
      return new Response(`Webhook signature verification failed: ${e.message}`, { status: 400 });
    }

    // We only need successful Checkout payments
    if (event.type === "checkout.session.completed") {
      const session = event.data.object;

      const mapSessionId = session?.metadata?.map_session_id;
      if (!mapSessionId) {
        return new Response("Missing metadata.map_session_id on session", { status: 400 });
      }

      const paidAt = new Date().toISOString();

      // Idempotent write: map_session_id is PRIMARY KEY
      await env.DB.prepare(
        `INSERT INTO entitlements (map_session_id, stripe_session_id, paid_at)
         VALUES (?, ?, ?)
         ON CONFLICT(map_session_id) DO UPDATE SET
           stripe_session_id=excluded.stripe_session_id,
           paid_at=excluded.paid_at`
      )
        .bind(mapSessionId, session.id, paidAt)
        .run();

      return new Response("ok", { status: 200 });
    }

    // For all other event types
    return new Response("ignored", { status: 200 });
  } catch (err) {
    return new Response(`stripe-webhook error:\n${err?.message || String(err)}`, { status: 500 });
  }
}
