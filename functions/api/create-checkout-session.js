import Stripe from "stripe";

export async function onRequestPost({ request, env }) {
  try {
    if (!env.STRIPE_SECRET_KEY) {
      return new Response("Missing env.STRIPE_SECRET_KEY", { status: 500 });
    }

    const { map_session_id } = await request.json().catch(() => ({}));
    if (!map_session_id) {
      return new Response("Missing map_session_id", { status: 400 });
    }

    const stripe = new Stripe(env.STRIPE_SECRET_KEY);

    // Use the request origin so preview deployments (pages.dev) work automatically
    const origin = new URL(request.url).origin;

    const session = await stripe.checkout.sessions.create({
      mode: "payment",
      payment_method_types: ["card"],
      line_items: [
        {
          price_data: {
            currency: "cad",
            product_data: {
              name: "MyMapData – Point Export",
              description: "Export points from a georeferenced map",
            },
            unit_amount: 1200, // $12.00 CAD
          },
          quantity: 1,
        },
      ],

      // tie the payment to this browser tab/session
      metadata: { map_session_id },

      success_url: `${origin}/app.html?paid=1&map_session_id=${encodeURIComponent(map_session_id)}`,
      cancel_url: `${origin}/app.html?canceled=1&map_session_id=${encodeURIComponent(map_session_id)}`,
    });

    return new Response(JSON.stringify({ url: session.url }), {
      headers: { "Content-Type": "application/json",
                  "Cache-Control": "no-store"
      },
    });
  } catch (err) {
    return new Response(
      `create-checkout-session error:\n${err?.message || String(err)}`,
      { status: 500 }
    );
  }
}
