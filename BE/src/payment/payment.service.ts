import { Injectable } from '@nestjs/common';
import Stripe from 'stripe';
import { PrismaService } from '../prisma.service';

@Injectable()
export class PaymentService {
  private stripe: Stripe;

  constructor(
    private prisma: PrismaService,
  ) {
    this.stripe = new Stripe(
      process.env.STRIPE_SECRET_KEY
    );
  }

  async createCheckoutSession(appointmentId: string, userId: string) {
    // Get appointment details
    const appointment = await this.prisma.appointment.findUnique({
      where: { id: appointmentId },
      include: {
        Patient: true,
        Doctor: true,
        Schedule: true,
      },
    });

    if (!appointment) {
      throw new Error('Appointment not found');
    }

    // Create a checkout session
    const session = await this.stripe.checkout.sessions.create({
      payment_method_types: ['card'],
      line_items: [
        {
          price_data: {
            currency: 'usd',
            product_data: {
              name: `Appointment with Dr. ${appointment.Doctor.userName}`,
              description: `Medical consultation on ${new Date(
                appointment.Schedule.startTime,
              ).toLocaleString()}`,
            },
            unit_amount: 2000, // $20.00 in cents
          },
          quantity: 1,
        },
      ],
      mode: 'payment',
      success_url: `${process.env.FRONTEND_URL}/appointments`,
      cancel_url: `${process.env.FRONTEND_URL}/appointments`,
      metadata: {
        appointmentId: appointmentId,
        userId: userId,
      },
    });

    // // Create a payment record in the database
    await this.prisma.payment.create({
      data: {
        userId: userId,
        appointmentId: appointmentId,
        amount: 20.0,
        currency: 'USD',
        status: 'PENDING',
        paymentMethod: 'card',
        transactionId: session.id
      },
    });

    return { sessionId: session.id, url: session.url };
  }

  async getPaymentStatus(sessionId: string) {
    const payment = await this.prisma.payment.findUnique({
      where: { transactionId: sessionId },
    });
    
    return payment;
  }
}
