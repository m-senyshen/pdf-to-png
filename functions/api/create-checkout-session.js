import Stripe from "stripe";

const origin = env.SITE_URL;

export async function onRequestPost({ env }) {
  try {
    if (!env.STRIPE_SECRET_KEY) {
      return new Response("Missing env.STRIPE_SECRET_KEY", { status: 500 });
    }
    if (!env.SITE_URL) {
      return new Response("Missing env.SITE_URL", { status: 500 });
    }

    const stripe = new Stripe(env.STRIPE_SECRET_KEY);

    const session = await stripe.checkout.sessions.create({
      mode: "payment",
      payment_method_types: ["card"],
      line_items: [
        {
          price_data: {
            currency: "cad",
            product_data: {
              name: "Map Extract – Point Export",
              description: "Export points from a georeferenced map",
            },
            unit_amount: 1200,
          },
          quantity: 1,
        },
      ],
      
      success_url: `${env.SITE_URL}/app.html?paid=1`,
      cancel_url: `${env.SITE_URL}/app.html?canceled=1`,
    });

    return new Response(JSON.stringify({ url: session.url }), {
      headers: { "Content-Type": "application/json" },
    });
  } catch (err) {
    // show the exact Stripe/CF error for debugging
    return new Response(
      `create-checkout-session error:\n${err?.message || String(err)}`,
      { status: 500 }
    );
  }
}
