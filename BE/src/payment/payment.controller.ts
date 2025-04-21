import {
  Body,
  Controller,
  Get,
  Headers,
  Param,
  Post,
  RawBodyRequest,
  Req,
  UseGuards,
} from '@nestjs/common';
import { JwtAuthGuard } from '../auth/@guard/jwt-auth.guard';
import { CurrentUser } from '../auth/@decorator/current-user.decorator';
import { PaymentService } from './payment.service';
import { Request } from 'express';
import { RolesGuard } from 'src/auth/@guard/roles.guard';
import Stripe from 'stripe';
import { Public } from 'src/auth/@decorator/public';

@Controller('payment')
// @UseGuards(JwtAuthGuard, RolesGuard)
export class PaymentController {
  constructor(private readonly paymentService: PaymentService) {}

  @Post('create-checkout-session/:appointmentId')
  async createCheckoutSession(
    @Param('appointmentId') appointmentId: string,
    @CurrentUser('userId') userId: string,
  ) {
    return this.paymentService.createCheckoutSession(appointmentId, userId);
  }

  @Public()
  @Post('webhook')
  async handleWebhook(@Req() request: RawBodyRequest<Request>, @Headers('stripe-signature') signature: string) {
    const payload = request.rawBody;    
    
    try {
      const stripe = new Stripe(process.env.STRIPE_SECRET_KEY);
      const event = stripe.webhooks.constructEvent(
        payload,
        signature,
        process.env.STRIPE_WEBHOOK_SECRET
      );
      
      await this.paymentService.handleWebhookEvent(event);
      return { received: true };
    } catch (err) {
      console.error(`Webhook Error: ${err.message}`);
      throw new Error(`Webhook Error: ${err.message}`);
    }
  }

  @Get('status/:sessionId')
  async getPaymentStatus(
    @Param('sessionId') sessionId: string,
  ) {
    return this.paymentService.getPaymentStatus(sessionId);
  }
}
