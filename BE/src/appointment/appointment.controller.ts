import {
  Body,
  Controller,
  Delete,
  Get,
  Param,
  Patch,
  Post,
  UseGuards,
} from '@nestjs/common';
import { User } from '@prisma/client';
import { CurrentUser } from '../auth/@decorator/current-user.decorator';
import { JwtAuthGuard } from '../auth/@guard/jwt-auth.guard';
import { RolesGuard } from '../auth/@guard/roles.guard';
import { AppointmentService } from './appointment.service';
import { CreateAppointmentDto } from './dto/create-appointment.dto';
import { UpdateAppointmentDto } from './dto/update-appointment.dto';

@Controller('appointments')
@UseGuards(JwtAuthGuard, RolesGuard)
export class AppointmentController {
  constructor(private readonly appointmentService: AppointmentService) {}

  @Post()
  async createAppointment(
    @Body() createAppointmentDto: CreateAppointmentDto,
    @CurrentUser('userId') userId: string,
  ) {
    return await this.appointmentService.createAppointment(
      createAppointmentDto,
      userId,
    );
  }

  @Get('user')
  async getUserAppointments(@CurrentUser() currentUser: User) {
    return await this.appointmentService.findUserAppointments(
      currentUser.id,
      currentUser.role,
    );
  }

  @Patch(':id')
  async updateAppointment(
    @Param('id') id: string,
    @Body() updateAppointmentDto: UpdateAppointmentDto,
    @CurrentUser() currentUser: User,
  ) {
    return await this.appointmentService.updateAppointment(
      id,
      updateAppointmentDto,
      currentUser,
    );
  }

  @Delete(':id')
  async cancelAppointment(
    @Param('id') id: string,
    @CurrentUser() currentUser: User,
  ) {
    return await this.appointmentService.cancelAppointment(id, currentUser);
  }
}
