import Stripe from "stripe";

export async function onRequestPost({ request, env }) {
  try {
    if (!env.STRIPE_SECRET_KEY) return new Response("Missing STRIPE_SECRET_KEY", { status: 500 });
    if (!env.STRIPE_WEBHOOK_SECRET) return new Response("Missing STRIPE_WEBHOOK_SECRET", { status: 500 });
    if (!env.DB) return new Response("Missing D1 binding DB", { status: 500 });

    // IMPORTANT: Stripe signature verification needs the *raw* body
    const body = await request.text();
    const sig = request.headers.get("stripe-signature");
    if (!sig) return new Response("Missing stripe-signature header", { status: 400 });

    const stripe = new Stripe(env.STRIPE_SECRET_KEY);

    let event;
    try {
      event = await stripe.webhooks.constructEventAsync(body, sig, env.STRIPE_WEBHOOK_SECRET);
    } catch (err) {
      return new Response(`Webhook signature verification failed: ${err?.message || err}`, { status: 400 });
    }

    // We care about successful checkout completion
    if (event.type === "checkout.session.completed") {
      const session = event.data.object;

      const map_session_id = session?.metadata?.map_session_id;
      if (!map_session_id) {
        return new Response("checkout.session.completed missing metadata.map_session_id", { status: 400 });
      }

      // Write entitlement (idempotent upsert)
      await env.DB.prepare(`
        INSERT INTO entitlements (
          session_id,
          paid,
          stripe_checkout_session_id,
          stripe_payment_intent_id,
          customer_email,
          updated_at
        ) VALUES (?, 1, ?, ?, ?, datetime('now'))
        ON CONFLICT(session_id) DO UPDATE SET
          paid = 1,
          stripe_checkout_session_id = excluded.stripe_checkout_session_id,
          stripe_payment_intent_id = excluded.stripe_payment_intent_id,
          customer_email = excluded.customer_email,
          updated_at = datetime('now')
      `).bind(
        sessionId,
        checkoutSessionId,
        paymentIntentId,
        customerEmail
      ).run();
    }

    return new Response("ok", { status: 200 });
  } catch (err) {
    return new Response(`stripe-webhook error: ${err?.message || String(err)}`, { status: 500 });
  }
}
