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

@Controller('payment')
@UseGuards(JwtAuthGuard, RolesGuard)
export class PaymentController {
  constructor(private readonly paymentService: PaymentService) {}

  @Post('create-checkout-session/:appointmentId')
  async createCheckoutSession(
    @Param('appointmentId') appointmentId: string,
    @CurrentUser('userId') userId: string,
  ) {
    console.log('first')
    return this.paymentService.createCheckoutSession(appointmentId, userId);
  }

  @Get('status/:sessionId')
  async getPaymentStatus(
    @Param('sessionId') sessionId: string,
  ) {
    return this.paymentService.getPaymentStatus(sessionId);
  }
}
