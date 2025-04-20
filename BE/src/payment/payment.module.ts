import { Module } from '@nestjs/common';
import { JwtService } from '@nestjs/jwt';
import { PaymentController } from './payment.controller';
import { PaymentService } from './payment.service';
import { PrismaService } from '../prisma.service';

@Module({
  controllers: [PaymentController],
  providers: [PaymentService, PrismaService, JwtService],
  exports: [PaymentService],
})
export class PaymentModule {}
