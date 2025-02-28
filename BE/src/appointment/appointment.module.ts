import { Module } from '@nestjs/common';
import { AppointmentController } from './appointment.controller';
import { AppointmentService } from './appointment.service';
import { PrismaService } from '../prisma.service';
import { JwtService } from '@nestjs/jwt';

@Module({
  controllers: [AppointmentController],
  providers: [AppointmentService, PrismaService, JwtService],
  exports: [AppointmentService],
})
export class AppointmentModule {}
