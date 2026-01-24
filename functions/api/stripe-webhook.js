import Stripe from "stripe";

export async function onRequestPost({ request, env }) {
  try {
    if (!env.STRIPE_WEBHOOK_SECRET) {
      return new Response("Missing env.STRIPE_WEBHOOK_SECRET", { status: 500 });
    }
    if (!env.STRIPE_SECRET_KEY) {
      return new Response("Missing env.STRIPE_SECRET_KEY", { status: 500 });
    }
    if (!env.DB) {
      return new Response("Missing D1 binding env.DB", { status: 500 });
    }

    const stripe = new Stripe(env.STRIPE_SECRET_KEY);

    const sig = request.headers.get("stripe-signature");
    if (!sig) return new Response("Missing stripe-signature header", { status: 400 });

    const body = await request.text();

    // IMPORTANT for Cloudflare: use async signature verification
    const event = await stripe.webhooks.constructEventAsync(
      body,
      sig,
      env.STRIPE_WEBHOOK_SECRET
    );

    // We only care about successful Checkout completion
    if (event.type === "checkout.session.completed") {
      const session = event.data.object;

      // browser-tab/session identifier from app.html
      const map_session_id = session?.metadata?.map_session_id;

      if (!map_session_id) {
        return new Response(
          "Missing session.metadata.map_session_id (did create-checkout-session send metadata?)",
          { status: 400 }
        );
      }

      const stripe_checkout_session_id = session.id ?? null;            // cs_test_...
      const stripe_payment_intent_id = session.payment_intent ?? null;  // pi_...
      const customer_email =
        session.customer_details?.email ?? session.customer_email ?? null;

      // UPSERT entitlement
      await env.DB.prepare(
        `
        INSERT INTO entitlements (
          map_session_id,
          paid,
          stripe_checkout_session_id,
          stripe_payment_intent_id,
          customer_email,
          paid_at,
          updated_at
        )
        VALUES (?, 1, ?, ?, ?, datetime('now'), datetime('now'))
        ON CONFLICT(map_session_id) DO UPDATE SET
          paid = 1,
          stripe_checkout_session_id = excluded.stripe_checkout_session_id,
          stripe_payment_intent_id = excluded.stripe_payment_intent_id,
          customer_email = COALESCE(excluded.customer_email, entitlements.customer_email),
          paid_at = COALESCE(entitlements.paid_at, excluded.paid_at),
          updated_at = datetime('now')
        `
      )
        .bind(
          map_session_id,
          stripe_checkout_session_id,
          stripe_payment_intent_id,
          customer_email
        )
        .run();
    }

    return new Response("ok", { status: 200 });
  } catch (err) {
    return new Response(`stripe-webhook error: ${err?.message || String(err)}`, {
      status: 500,
    });
  }
}
